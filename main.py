import requests
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.uix.button import Button

import arabic_reshaper
from bidi.algorithm import get_display

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

Builder.load_file('nobitexanalyzer.kv')

def reshape_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

MARKETS_URL = "https://api.wallex.ir/v1/markets"
OHLC_URL = "https://api.wallex.ir/v1/udf/history"

def get_active_markets():
    try:
        response = requests.get(MARKETS_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        markets = list(data['result']['symbols'].keys())
        return [m for m in markets if 'TMN' in m or 'USDT' in m]
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []

def get_market_history(symbol, resolution='1D', lookback_days=90):
    try:
        now = int(time.time())
        from_ts = now - (lookback_days * 86400)
        symbol_fmt = symbol.replace('-', '')
        params = {
            'symbol': symbol_fmt,
            'resolution': resolution,
            'from': from_ts,
            'to': now
        }
        response = requests.get(OHLC_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('s') == 'ok' and data.get('c'):
            df = pd.DataFrame(data)
            df[['o', 'h', 'l', 'c', 'v']] = df[['o', 'h', 'l', 'c', 'v']].astype(float)
            df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df.set_index('timestamp')
        return None
    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        return None

def analyze_market(df):
    if df is None or len(df) < 15:
        return None, "داده کافی نیست", {}

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    last_rsi = rsi.iloc[-1]

    signal = "نگهداری"
    targets = {}
    if last_rsi < 35:
        signal = "خرید"
        prev = df.iloc[-2]
        pivot = (prev['high'] + prev['low'] + prev['close']) / 3
        targets = {
            'R1': (2 * pivot) - prev['low'],
            'R2': pivot + (prev['high'] - prev['low']),
        }
    return signal, f"RSI: {last_rsi:.2f}", targets

def generate_price_chart(df, symbol):
    try:
        if df is None or df.empty:
            return None

        plt.figure(figsize=(6, 3))
        df['close'].plot(title=f"{symbol} - قیمت ۹۰ روز گذشته", color='blue')
        plt.xlabel("تاریخ")
        plt.ylabel("قیمت")
        plt.tight_layout()

        filename = f"chart_{symbol}.png"
        plt.savefig(filename)
        plt.close()
        return filename
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.markets_to_analyze = []
        self.analysis_results = []
        self.ids.title_label.text = reshape_text("تحلیلگر صرافی والکس")
        self.ids.analyze_button.text = reshape_text("شروع تحلیل")
        self.ids.status_label.text = reshape_text("برای شروع دکمه را بزنید")
        self.ids.creator_label.text = reshape_text("ساخته شده توسط دانیال بهساز")

    def start_analysis(self):
        self.analysis_results = []
        self.ids.results_grid.clear_widgets()
        self.ids.status_label.text = reshape_text("در حال دریافت بازارها...")
        Clock.schedule_once(self.fetch_markets)

    def fetch_markets(self, dt):
        self.markets_to_analyze = get_active_markets()
        if not self.markets_to_analyze:
            self.ids.status_label.text = reshape_text("خطا در دریافت بازارها")
            return
        Clock.schedule_once(self.analyze_next_market)

    def analyze_next_market(self, dt):
        if not self.markets_to_analyze:
            self.ids.status_label.text = reshape_text("تحلیل تمام شد")
            self.show_sorted_results()
            return

        symbol = self.markets_to_analyze.pop(0)
        self.ids.status_label.text = reshape_text(f"در حال تحلیل: {symbol}")

        df = get_market_history(symbol)
        signal, reason, targets = analyze_market(df)

        if signal == "خرید":
            price = df['close'].iloc[-1]
            r1 = targets.get('R1', 0)
            r2 = targets.get('R2', 0)
            profit_r1 = ((r1 - price) / price) * 100 if price else 0
            profit_r2 = ((r2 - price) / price) * 100 if price else 0
            chart_file = generate_price_chart(df, symbol)

            self.analysis_results.append({
                'symbol': symbol,
                'price': price,
                'r1': r1,
                'r2': r2,
                'profit_r1': profit_r1,
                'profit_r2': profit_r2,
                'reason': reason,
                'chart': chart_file
            })

        Clock.schedule_once(self.analyze_next_market)

    def show_sorted_results(self):
        self.ids.results_grid.clear_widgets()

        sorted_results = sorted(self.analysis_results, key=lambda x: x['profit_r1'], reverse=True)
        for res in sorted_results:
            text = (
                f"[b]{res['symbol']}[/b]\n"
                f"💰 قیمت فعلی: {res['price']}\n"
                f"🎯 هدف اول: {res['r1']} | سود: {res['profit_r1']:.2f}٪\n"
                f"🎯 هدف دوم: {res['r2']} | سود: {res['profit_r2']:.2f}٪\n"
                f"📌 دلیل: {res['reason']}"
            )
            label = Label(
                text=reshape_text(text),
                markup=True,
                font_name="Vazirmatn-Regular.ttf",
                halign='right',
                size_hint_y=None,
                height=180,
                text_size=(self.width - 20, None)
            )
            self.ids.results_grid.add_widget(label)

            if res['chart'] and os.path.exists(res['chart']):
                img = Image(source=res['chart'], size_hint_y=None, height=200)
                self.ids.results_grid.add_widget(img)

        save_btn = Button(
            text=reshape_text("📄 ذخیره به صورت PDF"),
            size_hint_y=None,
            height=50,
            on_press=self.ask_to_save_pdf
        )
        self.ids.results_grid.add_widget(save_btn)

    def ask_to_save_pdf(self, *args):
        save_path = "/storage/emulated/0/Download/analysis_report.pdf"
        self.generate_pdf(save_path)

    def generate_pdf(self, filename):
        font_path = "Vazirmatn-Regular.ttf"
        if not os.path.exists(font_path):
            self.ids.status_label.text = reshape_text("خطا: فایل فونت Vazirmatn-Regular.ttf یافت نشد.")
            return

        try:
            pdfmetrics.registerFont(TTFont('Vazir', font_path))
        except Exception as e:
            self.ids.status_label.text = reshape_text(f"خطا در ثبت فونت: {e}")
            return

        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        y = height - 40
        c.setFont("Vazir", 12)

        for res in self.analysis_results:
            lines = [
                f"{res['symbol']}",
                f"قیمت: {res['price']} تومان",
                f"هدف 1: {res['r1']} | سود: {res['profit_r1']:.2f}%",
                f"هدف 2: {res['r2']} | سود: {res['profit_r2']:.2f}%",
                f"دلیل: {res['reason']}",
                "-----------------------------"
            ]
            for line in lines:
                c.drawRightString(width - 40, y, reshape_text(line))
                y -= 20
                if y < 60:
                    c.showPage()
                    c.setFont("Vazir", 12)
                    y = height - 40
        c.save()
        self.ids.status_label.text = reshape_text(f"فایل PDF ذخیره شد: {filename}")

class NobitexAnalyzerApp(App):
    def build(self):
        self.title = "تحلیلگر والکس دانیال"
        return MainLayout()

if __name__ == '__main__':
    NobitexAnalyzerApp().run()