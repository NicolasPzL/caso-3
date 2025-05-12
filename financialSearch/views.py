from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

def user_login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        if not(username) or not(password):
            return render(request, 'login.html', {"err_msg":"Username and password required"})

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("/")
        else:
            return render(request, 'login.html', {"err_msg":"Username and/or password incorrect."})
    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    return redirect("/login")

@login_required(login_url='/login')
def home(request):
    return render(request, 'index.html')

@login_required(login_url='/login')
def getReturns(request):
    if request.method == 'POST':
        try:
            from_date = request.POST.get('from')
            to_date = request.POST.get('to')
            brand = request.POST.get('brand')

            # Log para depuración
            print(f"Received request - from: {from_date}, to: {to_date}, brand: {brand}")

            # Validate required fields
            if not all([from_date, to_date, brand]):
                return JsonResponse({
                    'error': 'Todos los campos son requeridos'
                }, status=400)

            # Validate dates
            try:
                start = datetime.strptime(from_date, '%Y-%m-%d')
                end = datetime.strptime(to_date, '%Y-%m-%d')
            except ValueError as e:
                print(f"Date parsing error: {str(e)}")  # Log para depuración
                return JsonResponse({
                    'error': 'Formato de fecha inválido. Use YYYY-MM-DD'
                }, status=400)

            # Check if dates are in the future
            current_date = datetime.now()
            if start > current_date or end > current_date:
                return JsonResponse({
                    'error': 'Las fechas no pueden ser en el futuro'
                }, status=400)

            # Check if start date is after end date
            if start > end:
                return JsonResponse({
                    'error': 'La fecha inicial debe ser anterior a la fecha final'
                }, status=400)

            # Get stock data
            try:
                ticker = yf.Ticker(brand)
                stock_data = ticker.history(start=start, end=end)
                print(f"Stock data shape: {stock_data.shape}")  # Log para depuración
            except Exception as e:
                print(f"Error fetching stock data: {str(e)}")  # Log para depuración
                return JsonResponse({
                    'error': f'Error al obtener datos de Yahoo Finance: {str(e)}'
                }, status=500)

            # Check if we got any data
            if stock_data.empty:
                return JsonResponse({
                    'error': f'No se encontraron datos para {brand} en el rango de fechas especificado'
                }, status=404)

            # Perform analysis
            try:
                analysis_results = analyze_data(stock_data)
            except Exception as e:
                print(f"Error in analysis: {str(e)}")  # Log para depuración
                return JsonResponse({
                    'error': f'Error en el análisis de datos: {str(e)}'
                }, status=500)
            
            # Prepare data for main chart
            try:
                closing_prices = stock_data['Close'].to_list()
                dates = stock_data.index.strftime('%Y-%m-%d').to_list()
                stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
                sma_5 = stock_data['SMA_5'].to_list()

                data = {
                    'brand': brand,
                    'data': [
                        {
                            'date': date,
                            'close': price,
                            'sma_5': sma if not pd.isna(sma) else None
                        } 
                        for date, price, sma in zip(dates, closing_prices, sma_5)
                    ],
                    'analysis': analysis_results
                }

                return JsonResponse(data)
            except Exception as e:
                print(f"Error preparing chart data: {str(e)}")  # Log para depuración
                return JsonResponse({
                    'error': f'Error al preparar los datos para el gráfico: {str(e)}'
                }, status=500)

        except Exception as e:
            print(f"Unexpected error: {str(e)}")  # Log para depuración
            return JsonResponse({
                'error': f'Error inesperado: {str(e)}'
            }, status=500)

    return JsonResponse({'error': 'Método no permitido'}, status=405)

def analyze_data(stock_data):
    """
    Realiza un análisis estadístico completo de los datos bursátiles.
    """
    if stock_data.empty:
        raise ValueError("No hay datos para analizar")

    df = stock_data.copy()
    
    try:
        # 1. Análisis de Retornos
        df['Daily_Return'] = df['Close'].pct_change() * 100
        df['Cumulative_Return'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100
        
        # 2. Medias Móviles
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_100'] = df['Close'].rolling(window=100).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # 3. Bandas de Bollinger
        df['Bollinger_Middle'] = df['SMA_20']
        df['Bollinger_STD'] = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = df['Bollinger_Middle'] + (df['Bollinger_STD'] * 2)
        df['Bollinger_Lower'] = df['Bollinger_Middle'] - (df['Bollinger_STD'] * 2)
        
        # 4. RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 5. MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 6. Volatilidad
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # 7. Volumen Analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # 8. Momentum Indicators
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['Rate_of_Change'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # 9. ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # 10. Análisis Predictivo Avanzado
        # Regresión lineal para predicción de precios
        x = np.arange(len(df))
        y = df['Close'].values
        slope, intercept = np.polyfit(x, y, 1)
        df['Linear_Prediction'] = slope * x + intercept
        
        # Predicción de tendencia usando múltiples indicadores
        df['Trend_Score'] = 0
        
        # RSI contribución
        df.loc[df['RSI'] < 30, 'Trend_Score'] += 1  # Sobreventa
        df.loc[df['RSI'] > 70, 'Trend_Score'] -= 1  # Sobrecompra
        
        # MACD contribución
        df.loc[df['MACD'] > df['MACD_Signal'], 'Trend_Score'] += 1  # Señal alcista
        df.loc[df['MACD'] < df['MACD_Signal'], 'Trend_Score'] -= 1  # Señal bajista
        
        # Bandas de Bollinger contribución
        df.loc[df['Close'] < df['Bollinger_Lower'], 'Trend_Score'] += 1  # Precio bajo
        df.loc[df['Close'] > df['Bollinger_Upper'], 'Trend_Score'] -= 1  # Precio alto
        
        # Volumen contribución
        df.loc[df['Volume_Ratio'] > 1.5, 'Trend_Score'] += 1  # Volumen alto
        df.loc[df['Volume_Ratio'] < 0.5, 'Trend_Score'] -= 1  # Volumen bajo
        
        # Momentum contribución
        df.loc[df['Momentum'] > 0, 'Trend_Score'] += 1  # Momentum positivo
        df.loc[df['Momentum'] < 0, 'Trend_Score'] -= 1  # Momentum negativo
        
        # Calcular el score final
        df['Final_Score'] = df['Trend_Score'].rolling(window=5).mean()
        
        # Función auxiliar para convertir NaN a None
        def clean_nan(x):
            return None if pd.isna(x) else x

        # Preparar datos para visualización
        analysis_data = {
            'text': generate_analysis_text(df),
            'price_data': {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'close': [clean_nan(x) for x in df['Close'].tolist()],
                'high': [clean_nan(x) for x in df['High'].tolist()],
                'low': [clean_nan(x) for x in df['Low'].tolist()],
                'volume': [clean_nan(x) for x in df['Volume'].tolist()]
            },
            'technical_indicators': {
                'moving_averages': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'sma_5': [clean_nan(x) for x in df['SMA_5'].tolist()],
                    'sma_20': [clean_nan(x) for x in df['SMA_20'].tolist()],
                    'sma_50': [clean_nan(x) for x in df['SMA_50'].tolist()],
                    'sma_100': [clean_nan(x) for x in df['SMA_100'].tolist()],
                    'ema_12': [clean_nan(x) for x in df['EMA_12'].tolist()],
                    'ema_26': [clean_nan(x) for x in df['EMA_26'].tolist()]
                },
                'bollinger_bands': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'middle': [clean_nan(x) for x in df['Bollinger_Middle'].tolist()],
                    'upper': [clean_nan(x) for x in df['Bollinger_Upper'].tolist()],
                    'lower': [clean_nan(x) for x in df['Bollinger_Lower'].tolist()]
                },
                'macd': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'macd': [clean_nan(x) for x in df['MACD'].tolist()],
                    'signal': [clean_nan(x) for x in df['MACD_Signal'].tolist()],
                    'histogram': [clean_nan(x) for x in df['MACD_Hist'].tolist()]
                },
                'rsi': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'values': [clean_nan(x) for x in df['RSI'].tolist()]
                },
                'momentum': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'values': [clean_nan(x) for x in df['Momentum'].tolist()],
                    'roc': [clean_nan(x) for x in df['Rate_of_Change'].tolist()]
                },
                'volatility': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'values': [clean_nan(x) for x in df['Volatility'].tolist()],
                    'atr': [clean_nan(x) for x in df['ATR'].tolist()]
                },
                'volume_analysis': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'volume': [clean_nan(x) for x in df['Volume'].tolist()],
                    'volume_sma': [clean_nan(x) for x in df['Volume_SMA'].tolist()],
                    'volume_ratio': [clean_nan(x) for x in df['Volume_Ratio'].tolist()]
                },
                'prediction': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'linear': [clean_nan(x) for x in df['Linear_Prediction'].tolist()],
                    'trend_score': [clean_nan(x) for x in df['Final_Score'].tolist()]
                }
            },
            'statistical_analysis': {
                'returns_distribution': {
                    'bins': [str(interval.mid) for interval in pd.cut(df['Daily_Return'], bins=20).value_counts().sort_index().index],
                    'values': pd.cut(df['Daily_Return'], bins=20).value_counts().sort_index().values.tolist()
                },
                'price_distribution': {
                    'bins': [str(interval.mid) for interval in pd.cut(df['Close'], bins=20).value_counts().sort_index().index],
                    'values': pd.cut(df['Close'], bins=20).value_counts().sort_index().values.tolist()
                }
            }
        }
        
        return analysis_data
        
    except Exception as e:
        raise ValueError(f"Error en el análisis de datos: {str(e)}")

def generate_analysis_text(df):
    """
    Genera un análisis textual detallado de los datos.
    """
    # Análisis de tendencia
    last_price = df['Close'].iloc[-1]
    last_sma_20 = df['SMA_20'].iloc[-1]
    last_sma_50 = df['SMA_50'].iloc[-1]
    last_sma_100 = df['SMA_100'].iloc[-1]
    
    # Determinar tendencia
    if last_price > last_sma_20 > last_sma_50 > last_sma_100:
        trend = "Fuertemente alcista"
    elif last_price > last_sma_20 and last_sma_20 > last_sma_50:
        trend = "Moderadamente alcista"
    elif last_price < last_sma_20 < last_sma_50 < last_sma_100:
        trend = "Fuertemente bajista"
    elif last_price < last_sma_20 and last_sma_20 < last_sma_50:
        trend = "Moderadamente bajista"
    else:
        trend = "Neutral"

    # Análisis RSI
    last_rsi = df['RSI'].iloc[-1]
    if last_rsi > 70:
        rsi_signal = "Sobrecomprado"
    elif last_rsi < 30:
        rsi_signal = "Sobreventa"
    else:
        rsi_signal = "Neutral"

    # Análisis MACD
    last_macd = df['MACD'].iloc[-1]
    last_signal = df['MACD_Signal'].iloc[-1]
    if last_macd > last_signal:
        macd_signal = "Alcista"
    else:
        macd_signal = "Bajista"

    # Análisis de Volatilidad
    current_volatility = df['Volatility'].iloc[-1]
    avg_volatility = df['Volatility'].mean()
    if current_volatility > avg_volatility * 1.5:
        volatility_signal = "Alta"
    elif current_volatility < avg_volatility * 0.5:
        volatility_signal = "Baja"
    else:
        volatility_signal = "Normal"

    # Análisis de Volumen
    last_volume_ratio = df['Volume_Ratio'].iloc[-1]
    if last_volume_ratio > 2:
        volume_signal = "Muy alto"
    elif last_volume_ratio > 1.5:
        volume_signal = "Alto"
    elif last_volume_ratio < 0.5:
        volume_signal = "Muy bajo"
    else:
        volume_signal = "Normal"

    # Análisis Predictivo
    final_score = df['Final_Score'].iloc[-1]
    if final_score > 2:
        recommendation = "COMPRAR"
        confidence = "Alta"
        reasoning = "Múltiples indicadores técnicos sugieren una fuerte tendencia alcista"
    elif final_score > 0:
        recommendation = "COMPRAR"
        confidence = "Moderada"
        reasoning = "Los indicadores técnicos sugieren una tendencia alcista moderada"
    elif final_score > -2:
        recommendation = "MANTENER"
        confidence = "Moderada"
        reasoning = "Los indicadores técnicos son mixtos, sugiriendo mantener posiciones actuales"
    else:
        recommendation = "VENDER"
        confidence = "Alta" if final_score < -3 else "Moderada"
        reasoning = "Los indicadores técnicos sugieren una tendencia bajista"

    # Generar texto de análisis
    analysis_text = f"""
    <h4>Análisis Técnico y Predictivo Completo</h4>
    
    <h5>Recomendación de Inversión</h5>
    <div class="recommendation-box">
        <h3 class="{'text-success' if recommendation == 'COMPRAR' else 'text-danger' if recommendation == 'VENDER' else 'text-warning'}">
            {recommendation}
        </h3>
        <p><strong>Nivel de Confianza:</strong> {confidence}</p>
        <p><strong>Razonamiento:</strong> {reasoning}</p>
    </div>

    <h5>Tendencia y Medias Móviles</h5>
    <ul>
        <li><strong>Tendencia actual:</strong> {trend}</li>
        <li><strong>Precio actual:</strong> ${last_price:.2f}</li>
        <li><strong>SMA 20:</strong> ${last_sma_20:.2f}</li>
        <li><strong>SMA 50:</strong> ${last_sma_50:.2f}</li>
        <li><strong>SMA 100:</strong> ${last_sma_100:.2f}</li>
    </ul>

    <h5>Indicadores de Momentum</h5>
    <ul>
        <li><strong>RSI (14):</strong> {last_rsi:.2f} - {rsi_signal}</li>
        <li><strong>MACD:</strong> {last_macd:.2f} - Señal {macd_signal}</li>
        <li><strong>Momentum (10):</strong> {df['Momentum'].iloc[-1]:.2f}</li>
        <li><strong>Rate of Change (10):</strong> {df['Rate_of_Change'].iloc[-1]:.2f}%</li>
    </ul>

    <h5>Volatilidad y Volumen</h5>
    <ul>
        <li><strong>Volatilidad actual:</strong> {volatility_signal} ({current_volatility:.2f}%)</li>
        <li><strong>ATR (14):</strong> ${df['ATR'].iloc[-1]:.2f}</li>
        <li><strong>Volumen:</strong> {volume_signal} ({last_volume_ratio:.2f}x promedio)</li>
    </ul>

    <h5>Estadísticas Descriptivas</h5>
    <ul>
        <li><strong>Retorno diario promedio:</strong> {df['Daily_Return'].mean():.2f}%</li>
        <li><strong>Desviación estándar diaria:</strong> {df['Daily_Return'].std():.2f}%</li>
        <li><strong>Retorno acumulado:</strong> {df['Cumulative_Return'].iloc[-1]:.2f}%</li>
        <li><strong>Precio máximo:</strong> ${df['High'].max():.2f}</li>
        <li><strong>Precio mínimo:</strong> ${df['Low'].min():.2f}</li>
    </ul>

    <h5>Análisis Predictivo Detallado</h5>
    <ul>
        <li><strong>Score de Tendencia:</strong> {final_score:.2f}</li>
        <li><strong>Tendencia de regresión:</strong> {"Alcista" if df['Linear_Prediction'].iloc[-1] > df['Linear_Prediction'].iloc[-2] else "Bajista"}</li>
        <li><strong>Proyección a corto plazo:</strong> ${df['Linear_Prediction'].iloc[-1]:.2f}</li>
        <li><strong>Potencial de movimiento:</strong> {abs(df['Linear_Prediction'].iloc[-1] - last_price) / last_price * 100:.2f}%</li>
    </ul>

    <h5>Análisis de Riesgo</h5>
    <ul>
        <li><strong>Nivel de Riesgo:</strong> {"Alto" if current_volatility > avg_volatility * 1.5 else "Bajo" if current_volatility < avg_volatility * 0.5 else "Moderado"}</li>
        <li><strong>Rango de Precios (ATR):</strong> ±${df['ATR'].iloc[-1]:.2f}</li>
        <li><strong>Probabilidad de Reversión:</strong> {"Alta" if abs(last_rsi - 50) > 20 else "Baja"}</li>
    </ul>
    """
    
    return analysis_text