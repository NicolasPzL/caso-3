from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

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



import yfinance as yf
from datetime import datetime
import pandas as pd

from datetime import datetime
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
import yfinance as yf
import pandas as pd

@login_required(login_url='/login')
def getReturns(request):
    if request.method == 'POST':

        from_date = request.POST.get('from')
        to_date = request.POST.get('to')
        brand = request.POST.get('brand')

        start = datetime.strptime(from_date, '%Y-%m-%d')
        end = datetime.strptime(to_date, '%Y-%m-%d')

        # Crear el objeto Ticker y usar .history()
        ticker = yf.Ticker(brand)
        stock_data = ticker.history(start=start, end=end)

        closing_prices = stock_data['Close'].to_list()
        dates = stock_data.index.strftime('%Y-%m-%d').to_list()

        stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
        sma_5 = stock_data['SMA_5'].to_list()

        analysis = analyze_data(sma_5)

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
            'analysis': analysis
        }

        return JsonResponse(data)

    return JsonResponse({'error': 'Método no permitido'}, status=405)


def analyze_data(prices):
    analysis = "Implemente su análisis acá."
    return analysis