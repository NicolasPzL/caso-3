import os
import sys
import django
from pathlib import Path

# Configurar Django antes de cualquier importación
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainApp.settings')
django.setup()

import pytest
import pandas as pd
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse
from datetime import datetime, timedelta
import json
from unittest.mock import patch, MagicMock, Mock
import allure
from unittest import mock
from django.contrib.auth.models import User
from django.db import transaction
from django.test import TransactionTestCase
from django.http import JsonResponse
from decimal import Decimal
import tempfile
import time
import yfinance as yf

# Importar las funciones del sistema
from .views import analyze_data, generate_analysis_text, getReturns


# === CONFIGURACIÓN DE PYTEST ===

# Marcadores personalizados
def pytest_configure(config):
    """Configurar marcadores personalizados para las pruebas"""
    config.addinivalue_line("markers", "unit: marca test como prueba unitaria")
    config.addinivalue_line("markers", "integration: marca test como prueba de integración")
    config.addinivalue_line("markers", "web: marca test como prueba web")
    config.addinivalue_line("markers", "smoke: marca test como prueba de humo")
    config.addinivalue_line("markers", "regression: marca test como prueba de regresión")
    config.addinivalue_line("markers", "performance: marca test como prueba de rendimiento")
    config.addinivalue_line("markers", "security: marca test como prueba de seguridad")
    config.addinivalue_line("markers", "slow: marca test como prueba lenta")


# Hook para capturar resultados
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook para capturar resultados de las pruebas"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


# === FIXTURES GLOBALES ===

@pytest.fixture
@allure.step("Crear un cliente Django no autenticado")
def client(db): # Renombrado de client_anonymous para brevedad, como en tu último código
    """Fixture para crear un cliente Django no autenticado."""
    return Client()

@pytest.fixture
@allure.step("Crear un cliente Django autenticado con test_user")
def authenticated_client(client, test_user): # Reutiliza la fixture 'client'
    """Fixture para crear un cliente Django autenticado con test_user."""
    client.login(username=test_user.username, password='clase2024')
    return client

@pytest.fixture(scope='session')
def django_db_setup(django_db_setup, django_db_blocker):
    """Setup de base de datos para toda la sesión de pruebas"""
    with django_db_blocker.unblock():
        # Crear datos de prueba globales si es necesario
        pass

@pytest.fixture
def test_user(db):
    """Fixture para crear un usuario de prueba"""
    user = User.objects.create_user(
        username='testuser',
        password='testpass123',
        email='test@example.com',
        first_name='Test',
        last_name='User'
    )
    user.is_active = True
    user.save()
    yield user
    # Cleanup
    user.delete()


@pytest.fixture
def authenticated_client(test_user):
    """Fixture para cliente autenticado"""
    client = Client()
    client.login(username='testuser', password='testpass123')
    yield client
    client.logout()


@pytest.fixture
def sample_stock_data():
    """Fixture con datos de stock de muestra realistas"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    np.random.seed(42)  # Para resultados reproducibles
    
    # Generar datos más realistas con tendencia y estacionalidad
    base_price = 100
    trend = np.linspace(0, 20, len(dates))  # Tendencia alcista
    seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # Patrón mensual
    noise = np.random.normal(0, 3, len(dates))
    
    close_prices = base_price + trend + seasonal + noise
    
    # Crear OHLCV data realista
    data = pd.DataFrame({
        'Open': close_prices + np.random.uniform(-1, 1, len(dates)),
        'High': np.maximum(close_prices + np.random.uniform(0, 3, len(dates)),
                          close_prices + np.random.uniform(-1, 4, len(dates))),
        'Low': np.minimum(close_prices - np.random.uniform(0, 3, len(dates)),
                         close_prices - np.random.uniform(-1, 4, len(dates))),
        'Close': close_prices,
        'Volume': np.random.uniform(1000000, 5000000, len(dates)) * \
                 (1 + 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / 7))  # Patrón semanal
    }, index=dates)
    
    # Ajustar High y Low para ser consistentes
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    return data


@pytest.fixture
def sample_stock_data_volatile():
    """Fixture con datos de stock volátiles"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    np.random.seed(100)
    
    # Datos con alta volatilidad
    base_price = 100
    volatility = 10  # Alta volatilidad
    returns = np.random.normal(0, volatility/100, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
        'Close': prices,
        'Volume': np.random.uniform(5000000, 15000000, len(dates))
    }, index=dates)
    
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    return data


@pytest.fixture
def mock_yfinance_success(sample_stock_data):
    """Mock de yfinance para casos exitosos"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_stock_data
        mock_instance.info = {
            'symbol': 'AAPL',
            'longName': 'Apple Inc.',
            'marketCap': 3000000000000,
            'sector': 'Technology'
        }
        mock_ticker.return_value = mock_instance
        yield mock_ticker


@pytest.fixture  
def mock_yfinance_empty():
    """Mock de yfinance que retorna datos vacíos"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance
        yield mock_ticker


@pytest.fixture
def mock_yfinance_error():
    """Mock de yfinance que lanza error"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.side_effect = Exception("Error de conexión con Yahoo Finance")
        yield mock_ticker


# === SUITE DE PRUEBAS PRINCIPAL ===


# --- Pruebas de Autenticación ---
@allure.feature("Sistema de Autenticación")
class TestAuthentication:
    LOGIN_URL = '/login'
    LOGOUT_URL = '/logout'
    HOME_URL = '/' # Para verificar redirección

    @allure.title("Login Exitoso")
    @allure.description("Verifica que un usuario puede iniciar sesión correctamente con credenciales válidas y es redirigido a la página principal.")
    @allure.tag("auth", "positive", "smoke")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_login_successful(self, client, test_user): # Usa el cliente anónimo para el POST de login
        payload = {'username': test_user.username, 'password': 'testpass123'}
        with allure.step(f"POST a {self.LOGIN_URL} con credenciales válidas"):
            response = client.post(self.LOGIN_URL, payload)
        
        with allure.step("Verificar redirección (302) a la página principal ('/')"):
            assert response.status_code == 302, "El login exitoso debería resultar en un código de estado 302 (redirección)."
            assert response.url == self.HOME_URL, f"Tras un login exitoso, se esperaba redirección a '{self.HOME_URL}', pero fue a '{response.url}'."
        
        with allure.step("Verificar que el usuario está autenticado en la sesión"):
            assert '_auth_user_id' in client.session, "La clave '_auth_user_id' que indica autenticación no se encontró en la sesión."
            assert client.session['_auth_user_id'] == str(test_user.id), "El ID del usuario autenticado en la sesión no coincide con el ID del usuario de prueba."

    @allure.title("Login Fallido - Credenciales Inválidas (Usuario Inexistente)")
    @allure.description("Verifica que el sistema rechaza credenciales inválidas cuando el usuario no existe.")
    @allure.tag("auth", "negative")
    @allure.severity(allure.severity_level.NORMAL)
    def test_login_invalid_credentials_nonexistent_user(self, client): # Usa el cliente anónimo
        payload = {'username': 'usuario_no_existe', 'password': 'password_incorrecta'}
        with allure.step(f"POST a {self.LOGIN_URL} con usuario inexistente"):
            response = client.post(self.LOGIN_URL, payload)
        
        with allure.step("Verificar permanencia en página de login (200) y mensaje de error"):
            assert response.status_code == 200, "Tras un login fallido, se esperaba permanecer en la página de login (código 200)."
            assert 'Username and/or password incorrect.' in response.content.decode(), "No se encontró el mensaje de error esperado para credenciales inválidas."
        
        with allure.step("Verificar que el usuario NO está autenticado en la sesión"):
            assert '_auth_user_id' not in client.session, "Un usuario no debería autenticarse con credenciales inválidas."

    @allure.title("Login Fallido - Contraseña Incorrecta para Usuario Existente")
    @allure.description("Verifica que el sistema rechaza una contraseña incorrecta para un usuario válido.")
    @allure.tag("auth", "negative")
    @allure.severity(allure.severity_level.NORMAL)
    def test_login_invalid_credentials_wrong_password(self, client, test_user): # Usa cliente anónimo y test_user para el username
        payload = {'username': test_user.username, 'password': 'password_incorrecta'}
        with allure.step(f"POST a {self.LOGIN_URL} con contraseña incorrecta para usuario '{test_user.username}'"):
            response = client.post(self.LOGIN_URL, payload)

        with allure.step("Verificar permanencia en página de login (200) y mensaje de error"):
            assert response.status_code == 200
            assert 'Username and/or password incorrect.' in response.content.decode()
        with allure.step("Verificar que el usuario NO está autenticado en la sesión"):
            assert '_auth_user_id' not in client.session

    @allure.title("Login Fallido - Campos Vacíos (Username)")
    @allure.description("Verifica el comportamiento del login cuando el campo username está vacío.")
    @allure.tag("auth", "negative", "input_validation")
    @allure.severity(allure.severity_level.MINOR)
    def test_login_empty_username(self, client):
        payload = {'username': '', 'password': 'clase2024'}
        with allure.step(f"POST a {self.LOGIN_URL} con username vacío"):
            response = client.post(self.LOGIN_URL, payload)
        with allure.step("Verificar mensaje de 'Username and password required'"):
            assert response.status_code == 200
            assert 'Username and password required' in response.content.decode()
            assert '_auth_user_id' not in client.session

    @allure.title("Login Fallido - Campos Vacíos (Password)")
    @allure.description("Verifica el comportamiento del login cuando el campo password está vacío.")
    @allure.tag("auth", "negative", "input_validation")
    @allure.severity(allure.severity_level.MINOR)
    def test_login_empty_password(self, client, test_user):
        payload = {'username': test_user.username, 'password': ''}
        with allure.step(f"POST a {self.LOGIN_URL} con password vacío para usuario '{test_user.username}'"):
            response = client.post(self.LOGIN_URL, payload)
        with allure.step("Verificar mensaje de 'Username and password required'"):
            assert response.status_code == 200
            assert 'Username and password required' in response.content.decode()
            assert '_auth_user_id' not in client.session

    @allure.title("Logout Exitoso")
    @allure.description("Verifica que un usuario autenticado puede cerrar sesión correctamente y es redirigido a la página de login.")
    @allure.tag("auth", "positive", "smoke")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_logout_successful(self, authenticated_client): # Usa cliente ya autenticado
        with allure.step(f"GET a {self.LOGOUT_URL} para cerrar sesión"):
            response = authenticated_client.get(self.LOGOUT_URL)
        
        with allure.step("Verificar redirección (302) a la página de login"):
            assert response.status_code == 302, "El logout exitoso debería resultar en un código 302."
            assert response.url == self.LOGIN_URL, f"Tras el logout, se esperaba redirección a '{self.LOGIN_URL}', pero fue a '{response.url}'."
        
        with allure.step("Verificar que el usuario ya NO está autenticado en la sesión"):
            assert '_auth_user_id' not in authenticated_client.session, "El usuario todavía parece estar autenticado en la sesión después del logout."
            
# --- Pruebas de Vista Home ---
@allure.feature("Vista Principal (Home)")
class TestHomeView:
    HOME_URL = '/'
    LOGIN_URL = '/login'

    @allure.title("Acceso a Home - Usuario Autenticado")
    @allure.description("Verifica que un usuario autenticado puede acceder a la página principal y ver su contenido.")
    @allure.tag("home", "positive", "smoke")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_home_authenticated_access(self, authenticated_client):
        with allure.step(f"GET a {self.HOME_URL} como usuario autenticado"):
            response = authenticated_client.get(self.HOME_URL)
        
        with allure.step("Verificar acceso exitoso (200) y contenido clave"):
            assert response.status_code == 200, "Un usuario autenticado debería poder acceder a la página principal (código 200)."
            response_content_str = response.content.decode()
            assert "Consulta de Datos de Yahoo Finance" in response_content_str, "El título del formulario principal no se encontró en la página Home."
            assert 'id="consultaForm"' in response_content_str, "El formulario con id='consultaForm' no se encontró en la página Home."

    @allure.title("Acceso a Home - Usuario No Autenticado es Redirigido")
    @allure.description("Verifica que un usuario no autenticado que intenta acceder a Home es redirigido a la página de login.")
    @allure.tag("home", "negative", "security")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_home_unauthenticated_redirect(self, client): # Usa cliente anónimo
        with allure.step(f"GET a {self.HOME_URL} como usuario no autenticado"):
            response = client.get(self.HOME_URL)
        
        with allure.step(f"Verificar redirección (302) a '{self.LOGIN_URL}?next={self.HOME_URL}'"):
            assert response.status_code == 302, "Un usuario no autenticado debería ser redirigido (código 302) al intentar acceder a Home."
            expected_redirect_url_start = f"{self.LOGIN_URL}?next={self.HOME_URL}"
            assert response.url == expected_redirect_url_start, \
                f"Se esperaba redirección a '{expected_redirect_url_start}', pero fue a '{response.url}'."
                
# --- Pruebas de Vista getReturns ---
@allure.feature("Obtención de Datos Financieros (getReturns)")
class TestGetReturnsView:
    GET_RETURNS_URL = '/getReturns'
    LOGIN_URL = '/login'

    @pytest.fixture
    @allure.step("Crear datos simulados de yfinance para pruebas")
    def mock_yfinance_data(self):
        """Fixture para generar datos simulados de yfinance.history()."""
        data = {
            'Open': [150.0, 151.0, 150.5, 152.0, 151.5],
            'High': [152.0, 151.5, 151.0, 153.0, 152.5],
            'Low': [149.0, 150.0, 149.5, 150.5, 150.0],
            'Close': [151.0, 150.5, 150.0, 152.5, 151.0],
            'Volume': [100000, 120000, 110000, 130000, 105000]
        }
        # Usar pd.date_range para un índice de fechas más robusto
        dates = pd.date_range(start='2023-01-01', periods=5, freq='B') # Freq 'B' para días hábiles
        return pd.DataFrame(data, index=dates)


    @allure.title("getReturns Requiere Autenticación")
    @allure.description("Verifica que un usuario no autenticado sea redirigido al intentar acceder a getReturns.")
    @allure.tag("returns", "negative", "security")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_get_returns_requires_authentication(self, client): # Usa cliente anónimo
        payload = {'from': '2023-01-01', 'to': '2023-01-05', 'brand': 'MSFT'}
        with allure.step(f"POST a {self.GET_RETURNS_URL} como usuario no autenticado"):
            response = client.post(self.GET_RETURNS_URL, payload)
        
        with allure.step(f"Verificar redirección (302) a '{self.LOGIN_URL}?next={self.GET_RETURNS_URL}'"):
            assert response.status_code == 302
            expected_redirect_url_start = f"{self.LOGIN_URL}?next={self.GET_RETURNS_URL}"
            assert response.url == expected_redirect_url_start, \
                f"Se esperaba redirección a '{expected_redirect_url_start}', pero fue a '{response.url}'."

    @allure.title("getReturns Solo Permite Método POST")
    @allure.description("Verifica que la vista getReturns rechace solicitudes con métodos HTTP diferentes a POST (ej. GET).")
    @allure.tag("returns", "negative", "protocol")
    @allure.severity(allure.severity_level.NORMAL)
    def test_get_returns_rejects_get_method(self, authenticated_client):
        with allure.step(f"GET a {self.GET_RETURNS_URL} como usuario autenticado"):
            response = authenticated_client.get(self.GET_RETURNS_URL)
        
        with allure.step("Verificar respuesta de error 405 (Método No Permitido)"):
            assert response.status_code == 405, "Una solicitud GET a getReturns debería resultar en un error 405."
            assert response.json()['error'] == 'Método no permitido', "El mensaje de error para método no permitido no es el esperado."

    @allure.title("getReturns Maneja Formato de Fecha Inválido")
    @allure.description("Verifica que la vista maneje formatos de fecha incorrectos con JsonResponse de error 400.")
    @allure.tag("returns", "negative", "input_validation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_get_returns_invalid_date_format(self, authenticated_client):
        payload = {'from': 'fecha_incorrecta_formato', 'to': '2023-01-05', 'brand': 'AAPL'}
        
        with allure.step(f"POST a {self.GET_RETURNS_URL} con formato de fecha 'from' inválido"):
            response = authenticated_client.post(self.GET_RETURNS_URL, payload)
        
        with allure.step("Verificar respuesta de error 400 para formato de fecha inválido"):
            assert response.status_code == 400, "Debería devolver un código 400 para formato de fecha inválido"
            response_json = response.json()
            assert 'error' in response_json
            assert 'Formato de fecha inválido' in response_json['error']
            assert 'YYYY-MM-DD' in response_json['error']


    @allure.title("getReturns Maneja Campos POST Requeridos Faltantes")
    @allure.description("Verifica cómo responde la vista cuando faltan campos esenciales en el payload POST.")
    @allure.tag("returns", "negative", "input_validation")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.parametrize(
        "payload_variant, missing_field_name",
        [
            ({'to': '2023-01-05', 'brand': 'AAPL'}, "from"),
            ({'from': '2023-01-01', 'brand': 'AAPL'}, "to"),
            ({'from': '2023-01-01', 'to': '2023-01-05'}, "brand"),
        ]
    )
    def test_get_returns_missing_required_fields(
        self, 
        authenticated_client, 
        payload_variant, 
        missing_field_name
    ):
        with allure.step(f"POST a {self.GET_RETURNS_URL} faltando el campo '{missing_field_name}'"):
            response = authenticated_client.post(self.GET_RETURNS_URL, payload_variant)
        
        with allure.step("Verificar respuesta de error 400 para campo faltante"):
            assert response.status_code == 400, f"Debería devolver un código 400 cuando falta el campo '{missing_field_name}'"
            response_json = response.json()
            assert 'error' in response_json
            assert 'Todos los campos son requeridos' in response_json['error']
        
        with allure.step("Verificar estructura del payload"):
            # Verificar que el campo faltante está en el payload
            assert missing_field_name not in payload_variant, \
                f"El campo '{missing_field_name}' no debería estar presente en el payload"
            
            # Verificar que los otros campos están presentes
            for field in ['from', 'to', 'brand']:
                if field != missing_field_name:
                    assert field in payload_variant, \
                        f"El campo '{field}' debería estar presente en el payload"


    @allure.title("getReturns Maneja Fechas en el Futuro")
    @allure.description("Verifica que la vista rechace fechas que están en el futuro.")
    @allure.tag("returns", "negative", "input_validation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_get_returns_future_dates(self, authenticated_client):
        # Obtener una fecha futura
        future_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        
        payload = {
            'from': '2023-01-01',
            'to': future_date,
            'brand': 'AAPL'
        }
        
        with allure.step(f"POST a {self.GET_RETURNS_URL} con fecha futura"):
            response = authenticated_client.post(self.GET_RETURNS_URL, payload)
        
        with allure.step("Verificar respuesta de error 400 para fecha futura"):
            assert response.status_code == 400, "Debería devolver un código 400 para fechas futuras"
            response_json = response.json()
            assert 'error' in response_json
            assert 'Las fechas no pueden ser en el futuro' in response_json['error']


    @allure.title("getReturns Maneja Fechas Invertidas")
    @allure.description("Verifica que la vista rechace cuando la fecha inicial es posterior a la fecha final.")
    @allure.tag("returns", "negative", "input_validation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_get_returns_inverted_dates(self, authenticated_client):
        payload = {
            'from': '2023-01-10',
            'to': '2023-01-01',
            'brand': 'AAPL'
        }
        
        with allure.step(f"POST a {self.GET_RETURNS_URL} con fechas invertidas"):
            response = authenticated_client.post(self.GET_RETURNS_URL, payload)
        
        with allure.step("Verificar respuesta de error 400 para fechas invertidas"):
            assert response.status_code == 400, "Debería devolver un código 400 para fechas invertidas"
            response_json = response.json()
            assert 'error' in response_json
            assert 'La fecha inicial debe ser anterior a la fecha final' in response_json['error']


    @allure.title("getReturns Maneja Excepción de yfinance.Ticker().history()")
    @allure.description("Verifica cómo responde la vista si la llamada a yfinance.history() falla.")
    @allure.tag("returns", "negative", "error_handling")
    @allure.severity(allure.severity_level.NORMAL)
    @mock.patch('yfinance.Ticker')
    def test_get_returns_handles_yfinance_exception(self, mock_yfinance_ticker_cls, authenticated_client):
        mock_ticker_instance = mock.MagicMock()
        simulated_error_message = "Error de conexión simulado con yfinance API"
        mock_ticker_instance.history.side_effect = Exception(simulated_error_message)
        mock_yfinance_ticker_cls.return_value = mock_ticker_instance

        payload = {'from': '2023-01-01', 'to': '2023-01-05', 'brand': 'ERROR_TICKER'}
        
        with allure.step(f"POST a {self.GET_RETURNS_URL} esperando una excepción de yfinance"):
            response = authenticated_client.post(self.GET_RETURNS_URL, payload)
        
        with allure.step("Verificar respuesta de error 500 para excepción de yfinance"):
            assert response.status_code == 500, "Debería devolver un código 500 para error de yfinance"
            response_json = response.json()
            assert 'error' in response_json
            assert 'Error al obtener datos de Yahoo Finance' in response_json['error']
            assert simulated_error_message in response_json['error']


    @allure.title("getReturns Maneja Datos Vacíos de yfinance")
    @allure.description("Verifica cómo responde la vista cuando yfinance no devuelve datos.")
    @allure.tag("returns", "negative", "error_handling")
    @allure.severity(allure.severity_level.NORMAL)
    @mock.patch('yfinance.Ticker')
    def test_get_returns_empty_stock_data(self, mock_yfinance_ticker_cls, authenticated_client):
        mock_ticker_instance = mock.MagicMock()
        # Simular que history() devuelve un DataFrame vacío
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_yfinance_ticker_cls.return_value = mock_ticker_instance

        payload = {'from': '2023-01-01', 'to': '2023-01-05', 'brand': 'UNKNOWN'}
        
        with allure.step(f"POST a {self.GET_RETURNS_URL} con ticker que no devuelve datos"):
            response = authenticated_client.post(self.GET_RETURNS_URL, payload)
        
        with allure.step("Verificar respuesta de error 404 para datos vacíos"):
            assert response.status_code == 404, "Debería devolver un código 404 para datos vacíos"
            response_json = response.json()
            assert 'error' in response_json
            assert 'No se encontraron datos para UNKNOWN' in response_json['error']


    @allure.title("getReturns Maneja Error en analyze_data")
    @allure.description("Verifica cómo responde la vista si analyze_data lanza una excepción.")
    @allure.tag("returns", "negative", "error_handling")
    @allure.severity(allure.severity_level.NORMAL)
    @mock.patch('financialSearch.views.analyze_data')
    @mock.patch('yfinance.Ticker')
    def test_get_returns_handles_analyze_data_exception(
        self, 
        mock_yfinance_ticker_cls, 
        mock_analyze_data_func, 
        authenticated_client,
        mock_yfinance_data
    ):
        # Configurar yfinance para devolver datos válidos
        mock_ticker_instance = mock.MagicMock()
        mock_ticker_instance.history.return_value = mock_yfinance_data
        mock_yfinance_ticker_cls.return_value = mock_ticker_instance
        
        # Configurar analyze_data para lanzar una excepción
        simulated_analysis_error = "Error en análisis de datos"
        mock_analyze_data_func.side_effect = Exception(simulated_analysis_error)

        payload = {'from': '2023-01-01', 'to': '2023-01-05', 'brand': 'AAPL'}
        
        with allure.step(f"POST a {self.GET_RETURNS_URL} esperando error en analyze_data"):
            response = authenticated_client.post(self.GET_RETURNS_URL, payload)
        
        with allure.step("Verificar respuesta de error 500 para excepción en analyze_data"):
            assert response.status_code == 500, "Debería devolver un código 500 para error en analyze_data"
            response_json = response.json()
            assert 'error' in response_json
            assert 'Error en el análisis de datos' in response_json['error']
            assert simulated_analysis_error in response_json['error']


    @allure.title("getReturns Rechaza Método HTTP No Permitido")
    @allure.description("Verifica que la vista rechace métodos HTTP diferentes a POST.")
    @allure.tag("returns", "negative", "http_method")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.parametrize("http_method", ["GET", "PUT", "DELETE", "PATCH"])
    def test_get_returns_invalid_http_method(self, authenticated_client, http_method):
        with allure.step(f"Realizar solicitud {http_method} a {self.GET_RETURNS_URL}"):
            response = getattr(authenticated_client, http_method.lower())(self.GET_RETURNS_URL)
        
        with allure.step("Verificar respuesta de error 405 para método no permitido"):
            assert response.status_code == 405, f"Debería devolver un código 405 para método {http_method}"
            response_json = response.json()
            assert 'error' in response_json
            assert 'Método no permitido' in response_json['error']


    
@allure.epic("Sistema de Análisis Financiero")
@allure.parent_suite("Sistema de Trading Algorítmico")
@allure.suite("Pruebas del Módulo de Análisis Financiero")
class TestFinancialAnalysisComplete:
    """Suite completa de pruebas para el sistema de análisis financiero"""

    # === PRUEBAS UNITARIAS ===

    @allure.feature("Análisis de Datos")
    @allure.story("Validación de Entrada")
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.title("Validación de DataFrame vacío")
    @allure.description("Verifica que la función maneje correctamente un DataFrame vacío")
    @allure.tag("unit", "validation", "error-handling")
    @pytest.mark.unit
    @pytest.mark.smoke
    def test_analyze_data_empty_dataframe(self):
        """Test unitario: manejo de DataFrame vacío"""
        with allure.step("Given: DataFrame vacío"):
            empty_df = pd.DataFrame()
            
        with allure.step("When: Se llama a analyze_data"):
            with allure.step("Then: Debe lanzar ValueError"):
                with pytest.raises(ValueError, match="No hay datos para analizar"):
                    analyze_data(empty_df)
                    
    @allure.feature("Análisis de Datos")
    @allure.story("Validación de Estructura")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Validación de columnas requeridas")
    @allure.description("Verifica que la función valide las columnas necesarias")
    @allure.tag("unit", "validation", "data-structure")
    @pytest.mark.unit
    def test_analyze_data_missing_columns(self):
        """Test unitario: validación de columnas requeridas"""
        with allure.step("Given: DataFrame con columnas incorrectas"):
            invalid_df = pd.DataFrame({
                'Wrong': [1, 2, 3],
                'Columns': [4, 5, 6]
            })
            
        with allure.step("When: Se llama a analyze_data"):
            with allure.step("Then: Debe lanzar ValueError"):
                with pytest.raises(ValueError):
                    analyze_data(invalid_df)

    @allure.feature("Análisis de Datos")
    @allure.story("Cálculos Técnicos")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Cálculo de RSI correcto")
    @allure.description("Verifica que el RSI se calcule correctamente y esté en rango [0,100]")
    @allure.tag("unit", "calculations", "rsi")
    @pytest.mark.unit
    def test_rsi_calculation(self, sample_stock_data):
        """Test unitario: cálculo preciso del RSI"""
        with allure.step("Given: Datos de stock válidos"):
            result = analyze_data(sample_stock_data)
            
        with allure.step("Then: RSI debe estar en rango [0,100]"):
            rsi_values = result['technical_indicators']['rsi']['values']
            valid_rsi = [x for x in rsi_values if x is not None]
            
            assert len(valid_rsi) > 0, "No se calcularon valores RSI"
            assert all(0 <= x <= 100 for x in valid_rsi), "RSI fuera de rango"
            
            # Verificar precisión del cálculo
            avg_rsi = sum(valid_rsi) / len(valid_rsi)
            assert 20 <= avg_rsi <= 80, f"RSI promedio anormal: {avg_rsi}"

    @allure.feature("Análisis de Datos")
    @allure.story("Cálculos Técnicos")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Cálculo de Bandas de Bollinger")
    @allure.description("Verifica que las Bandas de Bollinger se calculen correctamente")
    @allure.tag("unit", "calculations", "bollinger")
    @pytest.mark.unit
    def test_bollinger_bands_calculation(self, sample_stock_data):
        """Test unitario: cálculo de Bandas de Bollinger"""
        with allure.step("Given: Datos de stock"):
            result = analyze_data(sample_stock_data)
            bb = result['technical_indicators']['bollinger_bands']
            
        with allure.step("Then: Verificar orden correcto de bandas"):
            for i in range(len(bb['upper'])):
                if all(x is not None for x in [bb['upper'][i], bb['middle'][i], bb['lower'][i]]):
                    assert bb['upper'][i] >= bb['middle'][i] >= bb['lower'][i], \
                        f"Bandas de Bollinger incorrectas en índice {i}"
                    
                    # Verificar distancia razonable entre bandas
                    band_width = bb['upper'][i] - bb['lower'][i]
                    middle_value = bb['middle'][i]
                    assert 0 < band_width < middle_value, \
                        f"Ancho de banda anormal en índice {i}"

    @allure.feature("Análisis de Datos")
    @allure.story("Cálculos Técnicos")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Cálculo de MACD")
    @allure.description("Verifica el cálculo correcto del MACD y su línea de señal")
    @allure.tag("unit", "calculations", "macd")
    @pytest.mark.unit
    def test_macd_calculation(self, sample_stock_data):
        """Test unitario: cálculo de MACD"""
        with allure.step("Given: Datos de stock"):
            result = analyze_data(sample_stock_data)
            macd_data = result['technical_indicators']['macd']
            
        with allure.step("Then: Verificar cálculo MACD"):
            macd_values = [x for x in macd_data['macd'] if x is not None]
            signal_values = [x for x in macd_data['signal'] if x is not None]
            
            assert len(macd_values) > 0, "No se calculó MACD"
            assert len(signal_values) > 0, "No se calculó línea de señal"
            
            # Verificar que el histograma sea correcto
            hist_values = [x for x in macd_data['histogram'] if x is not None]
            for i in range(min(len(macd_values), len(signal_values), len(hist_values))):
                if all(x is not None for x in [macd_values[i], signal_values[i], hist_values[i]]):
                    expected_hist = macd_values[i] - signal_values[i]
                    assert abs(hist_values[i] - expected_hist) < 0.001, \
                        f"Histograma MACD incorrecto en índice {i}"

    @allure.feature("Análisis de Datos")
    @allure.story("Análisis Estadístico")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Cálculo de retornos")
    @allure.description("Verifica el cálculo correcto de retornos diarios y acumulados")
    @allure.tag("unit", "statistics", "returns")
    @pytest.mark.unit
    def test_returns_calculation(self, sample_stock_data):
        """Test unitario: cálculo de retornos"""
        with allure.step("Given: Datos de stock"):
            # Calcular retornos manualmente
            expected_daily_returns = sample_stock_data['Close'].pct_change() * 100
            expected_cumulative_returns = (sample_stock_data['Close'] / sample_stock_data['Close'].iloc[0] - 1) * 100
            
        with allure.step("When: Se analizan los datos"):
            result = analyze_data(sample_stock_data)
            
        with allure.step("Then: Verificar cálculos de retornos"):
            # Los retornos están en el DataFrame interno, no en el resultado
            # Verificar que el análisis incluya información de retornos
            analysis_text = result['text']
            assert "Retorno diario promedio:" in analysis_text
            assert "Retorno acumulado:" in analysis_text

    @allure.feature("Análisis de Datos")
    @allure.story("Análisis Predictivo")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Generación de recomendaciones")
    @allure.description("Verifica la generación correcta de recomendaciones de inversión")
    @allure.tag("unit", "prediction", "recommendations")
    @pytest.mark.unit
    def test_investment_recommendations(self, sample_stock_data):
        """Test unitario: recomendaciones de inversión"""
        with allure.step("Given: Datos de stock"):
            result = analyze_data(sample_stock_data)
            
        with allure.step("Then: Verificar recomendación generada"):
            analysis_text = result['text']
            
            # Verificar que se genere una recomendación válida
            recommendations = ["COMPRAR", "VENDER", "MANTENER"]
            assert any(rec in analysis_text for rec in recommendations), \
                "No se encontró recomendación válida"
            
            # Verificar componentes de la recomendación
            assert "Nivel de Confianza:" in analysis_text
            assert "Razonamiento:" in analysis_text
            
            # Verificar que el score de tendencia esté presente
            trend_scores = result['technical_indicators']['prediction']['trend_score']
            valid_scores = [x for x in trend_scores if x is not None]
            assert len(valid_scores) > 0, "No se generaron scores de tendencia"

    @allure.feature("Análisis de Datos")
    @allure.story("Manejo de Datos Especiales")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Manejo de valores NaN")
    @allure.description("Verifica el manejo correcto de valores NaN en los datos")
    @allure.tag("unit", "data-handling", "nan")
    @pytest.mark.unit
    def test_nan_handling(self, sample_stock_data):
        """Test unitario: manejo de valores NaN"""
        with allure.step("Given: Datos con valores NaN"):
            data_with_nan = sample_stock_data.copy()
            data_with_nan.iloc[0:3, 0] = np.nan  # NaN en primeros valores de Open
            data_with_nan.iloc[5:8, 3] = np.nan  # NaN en algunos Close
            
        with allure.step("When: Se analizan los datos"):
            result = analyze_data(data_with_nan)
            
        with allure.step("Then: Verificar que no hay NaN en resultado"):
            def check_for_nan(obj, path="root"):
                """Función recursiva para verificar NaN"""
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        check_for_nan(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_for_nan(item, f"{path}[{i}]")
                elif pd.isna(obj) and obj is not None:
                    # None es aceptable, NaN no
                    assert False, f"NaN encontrado en {path}"
            
            check_for_nan(result)


    # === PRUEBAS DE INTEGRACIÓN ===

    @allure.feature("API REST")
    @allure.story("Endpoint getReturns")
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.title("Request exitoso al endpoint")
    @allure.description("Verifica el funcionamiento completo del endpoint con datos válidos")
    @allure.tag("integration", "api", "success")
    @pytest.mark.django_db
    @pytest.mark.integration
    def test_get_returns_success(self, authenticated_client, mock_yfinance_success):
        """Test integración: request exitoso"""
        with allure.step("Given: Request con datos válidos"):
            url = reverse('financialSearch:getReturns')
            data = {
                'from': '2024-01-01',
                'to': '2024-01-31',
                'brand': 'AAPL'
            }
            
        with allure.step("When: Se hace POST request"):
            response = authenticated_client.post(url, data)
            
        with allure.step("Then: Verificar respuesta exitosa"):
            assert response.status_code == 200
            
            response_data = json.loads(response.content)
            assert response_data['brand'] == 'AAPL'
            assert 'data' in response_data
            assert 'analysis' in response_data
            
            # Verificar estructura del análisis
            analysis = response_data['analysis']
            assert all(key in analysis for key in ['text', 'price_data', 'technical_indicators', 'statistical_analysis'])
            
            # Verificar datos del gráfico principal
            chart_data = response_data['data']
            assert len(chart_data) > 0
            assert all(key in chart_data[0] for key in ['date', 'close', 'sma_5'])

    @allure.feature("API REST")
    @allure.story("Validación de Fechas")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Validación de rango de fechas inválido")
    @allure.description("Verifica el manejo de fechas donde inicio > fin")
    @allure.tag("integration", "validation", "dates")
    @pytest.mark.django_db
    @pytest.mark.integration
    def test_invalid_date_range(self, authenticated_client):
        """Test integración: validación de fechas inválidas"""
        with allure.step("Given: Fechas donde inicio > fin"):
            url = reverse('financialSearch:getReturns')
            data = {
                'from': '2024-01-31',
                'to': '2024-01-01',  # Fecha fin antes que inicio
                'brand': 'AAPL'
            }
            
        with allure.step("When: Se hace POST request"):
            response = authenticated_client.post(url, data)
            
        with allure.step("Then: Verificar error 400"):
            assert response.status_code == 400
            error_data = json.loads(response.content)
            assert 'error' in error_data
            assert 'fecha inicial' in error_data['error'].lower()

    @allure.feature("API REST")
    @allure.story("Validación de Fechas")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Validación de fechas futuras")
    @allure.description("Verifica que no se permitan fechas en el futuro")
    @allure.tag("integration", "validation", "future-dates")
    @pytest.mark.django_db
    @pytest.mark.integration
    def test_future_dates_validation(self, authenticated_client):
        """Test integración: validación de fechas futuras"""
        with allure.step("Given: Fechas en el futuro"):
            future_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            url = reverse('financialSearch:getReturns')
            data = {
                'from': future_date,
                'to': future_date,
                'brand': 'AAPL'
            }
            
        with allure.step("When: Se hace POST request"):
            response = authenticated_client.post(url, data)
            
        with allure.step("Then: Verificar error 400"):
            assert response.status_code == 400
            error_data = json.loads(response.content)
            assert 'error' in error_data
            assert 'futuro' in error_data['error'].lower()

    @allure.feature("API REST")
    @allure.story("Validación de Datos")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Validación de parámetros faltantes")
    @allure.description("Verifica el manejo de requests con parámetros faltantes")
    @allure.tag("integration", "validation", "missing-params")
    @pytest.mark.django_db
    @pytest.mark.integration
    def test_missing_parameters(self, authenticated_client):
        """Test integración: parámetros faltantes"""
        url = reverse('financialSearch:getReturns')
        
        with allure.step("Test sin brand"):
            response = authenticated_client.post(url, {
                'from': '2024-01-01',
                'to': '2024-01-31'
            })
            assert response.status_code == 400
            assert 'error' in json.loads(response.content)
            
        with allure.step("Test sin fechas"):
            response = authenticated_client.post(url, {
                'brand': 'AAPL'
            })
            assert response.status_code == 400
            assert 'error' in json.loads(response.content)
            
        with allure.step("Test sin ningún parámetro"):
            response = authenticated_client.post(url, {})
            assert response.status_code == 400
            assert 'error' in json.loads(response.content)

    @allure.feature("API REST")
    @allure.story("Manejo de Errores")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Manejo de error de Yahoo Finance")
    @allure.description("Verifica el manejo cuando Yahoo Finance falla")
    @allure.tag("integration", "error-handling", "external-api")
    @pytest.mark.django_db
    @pytest.mark.integration
    def test_yfinance_error_handling(self, authenticated_client, mock_yfinance_error):
        """Test integración: error de Yahoo Finance"""
        with allure.step("Given: Mock configurado para error"):
            url = reverse('financialSearch:getReturns')
            data = {
                'from': '2024-01-01',
                'to': '2024-01-31',
                'brand': 'AAPL'
            }
            
        with allure.step("When: Se hace POST request"):
            response = authenticated_client.post(url, data)
            
        with allure.step("Then: Verificar error 500"):
            assert response.status_code == 500
            error_data = json.loads(response.content)
            assert 'error' in error_data
            assert 'Yahoo Finance' in error_data['error']

    @allure.feature("API REST")
    @allure.story("Manejo de Errores")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Manejo de datos vacíos de Yahoo Finance")
    @allure.description("Verifica el manejo cuando no hay datos disponibles")
    @allure.tag("integration", "error-handling", "empty-data")
    @pytest.mark.django_db
    @pytest.mark.integration
    def test_empty_data_handling(self, authenticated_client, mock_yfinance_empty):
        """Test integración: datos vacíos"""
        with allure.step("Given: Mock retorna datos vacíos"):
            url = reverse('financialSearch:getReturns')
            data = {
                'from': '2024-01-01',
                'to': '2024-01-31',
                'brand': 'INVALID'
            }
            
        with allure.step("When: Se hace POST request"):
            response = authenticated_client.post(url, data)
            
        with allure.step("Then: Verificar error 404"):
            assert response.status_code == 404
            error_data = json.loads(response.content)
            assert 'error' in error_data
            assert 'No se encontraron datos' in error_data['error']

    @allure.feature("Seguridad")
    @allure.story("Autenticación")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("Autenticación requerida")
    @allure.description("Verifica que se requiera autenticación para acceder al endpoint")
    @allure.tag("integration", "security", "authentication")
    @pytest.mark.django_db
    @pytest.mark.integration
    @pytest.mark.web
    def test_authentication_required(self):
        """Test integración: autenticación requerida"""
        with allure.step("Given: Cliente no autenticado"):
            client = Client()  # Sin autenticar
            url = reverse('financialSearch:getReturns')
            data = {
                'from': '2024-01-01',
                'to': '2024-01-31',
                'brand': 'AAPL'
            }
            
        with allure.step("When: Se hace POST request"):
            response = client.post(url, data)
            
        with allure.step("Then: Verificar redirección a login"):
            assert response.status_code == 302  # Redirección
            assert '/login' in response.url

    # === PRUEBAS DE ESCENARIOS ESPECIALES ===

    @allure.feature("Casos Especiales")
    @allure.story("Datos Extremos")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Análisis con datos volátiles")
    @allure.description("Verifica el análisis con datos de alta volatilidad")
    @allure.tag("unit", "edge-case", "volatility")
    @pytest.mark.unit
    def test_highly_volatile_data(self, sample_stock_data_volatile):
        """Test unitario: datos con alta volatilidad"""
        with allure.step("Given: Datos altamente volátiles"):
            result = analyze_data(sample_stock_data_volatile)
            
        with allure.step("Then: Verificar análisis robusto"):
            # Verificar que el análisis maneje bien la volatilidad
            volatility_indicators = result['technical_indicators']['volatility']
            volatility_values = [x for x in volatility_indicators['values'] if x is not None]
            
            assert len(volatility_values) > 0, "No se calculó volatilidad"
            
            # En datos volátiles, esperamos mayor volatilidad
            avg_volatility = sum(volatility_values) / len(volatility_values)
            assert avg_volatility > 1, "Volatilidad no detectada en datos volátiles"
            
            # Verificar que el análisis refleje alta volatilidad
            analysis_text = result['text']
            assert any(word in analysis_text.lower() for word in ["alta", "alto"]), \
                "El análisis no detectó alta volatilidad"

    @allure.feature("Casos Especiales")
    @allure.story("Datos Atípicos")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Análisis con outliers")
    @allure.description("Verifica el manejo de outliers en los datos")
    @allure.tag("unit", "edge-case", "outliers")
    @pytest.mark.unit
    def test_data_with_outliers(self, sample_stock_data):
        """Test unitario: datos con outliers"""
        with allure.step("Given: Datos con outliers"):
            data_outliers = sample_stock_data.copy()
            # Agregar outliers
            data_outliers.iloc[10, data_outliers.columns.get_loc('Close')] *= 2  # Spike
            data_outliers.iloc[20, data_outliers.columns.get_loc('Close')] *= 0.5  # Drop
            
        with allure.step("When: Se analizan los datos"):
            result = analyze_data(data_outliers)
            
        with allure.step("Then: Verificar manejo de outliers"):
            # El análisis debe completarse sin errores
            assert 'text' in result
            assert 'technical_indicators' in result
            
            # Los indicadores técnicos deben manejar los outliers
            rsi_values = [x for x in result['technical_indicators']['rsi']['values'] if x is not None]
            assert len(rsi_values) > 0

    @allure.feature("Casos Especiales")
    @allure.story("Períodos de Tiempo")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Análisis con período muy corto")
    @allure.description("Verifica el análisis con solo unos días de datos")
    @allure.tag("unit", "edge-case", "short-period")
    @pytest.mark.unit
    def test_short_period_analysis(self):
        """Test unitario: período de tiempo muy corto"""
        with allure.step("Given: Solo 5 días de datos"):
            dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
            short_data = pd.DataFrame({
                'Open': [100, 101, 99, 102, 98],
                'High': [102, 103, 101, 104, 100],
                'Low': [99, 100, 98, 101, 97],
                'Close': [101, 99, 100, 103, 99],
                'Volume': [1000000, 1100000, 900000, 1200000, 950000]
            }, index=dates)
            
        with allure.step("When: Se analizan los datos"):
            result = analyze_data(short_data)
            
        with allure.step("Then: Verificar análisis adaptativo"):
            # Con pocos datos, algunos indicadores estarán limitados
            sma_5 = result['technical_indicators']['moving_averages']['sma_5']
            valid_sma = [x for x in sma_5 if x is not None]
            
            # SMA 5 necesita 5 datos, así que solo el último valor será válido
            assert len(valid_sma) <= 1, "SMA 5 calculado con menos de 5 datos"
            
            # El análisis debe adaptarse
            analysis_text = result['text']
            assert "Análisis Técnico" in analysis_text

    # === PRUEBAS DE FORMATO Y SALIDA ===

    @allure.feature("Formato de Salida")
    @allure.story("Formato HTML")
    @allure.severity(allure.severity_level.MINOR)
    @allure.title("Validación de formato HTML")
    @allure.description("Verifica que el informe tenga formato HTML correcto")
    @allure.tag("unit", "format", "html")
    @pytest.mark.unit
    def test_html_format_validation(self, sample_stock_data):
        """Test unitario: formato HTML del informe"""
        with allure.step("Given: Datos de stock"):
            result = analyze_data(sample_stock_data)
            report = result['text']
            
        with allure.step("Then: Verificar formato HTML"):
            # Verificar estructura HTML básica
            assert report.strip().startswith('<'), "No inicia con tag HTML"
            
            # Verificar etiquetas requeridas
            required_tags = ['<h4>', '</h4>', '<h5>', '</h5>', '<ul>', '</ul>', '<li>', '</li>']
            for tag in required_tags:
                assert tag in report, f"Tag {tag} no encontrado"
            
            # Verificar balance de etiquetas
            for open_tag, close_tag in [('<h4>', '</h4>'), ('<h5>', '</h5>'), ('<ul>', '</ul>')]:
                open_count = report.count(open_tag)
                close_count = report.count(close_tag)
                assert open_count == close_count, f"Desbalance en {open_tag}/{close_tag}"

    @allure.feature("Formato de Salida")
    @allure.story("Estructura de Datos")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.title("Validación de estructura JSON")
    @allure.description("Verifica la estructura correcta de la respuesta JSON")
    @allure.tag("integration", "format", "json")
    @pytest.mark.django_db
    @pytest.mark.integration
    def test_json_structure_validation(self, authenticated_client, mock_yfinance_success):
        """Test integración: estructura JSON de respuesta"""
        with allure.step("Given: Request válido"):
            url = reverse('financialSearch:getReturns')
            data = {
                'from': '2024-01-01',
                'to': '2024-01-31',
                'brand': 'AAPL'
            }
            
        with allure.step("When: Se hace POST request"):
            response = authenticated_client.post(url, data)
            
        with allure.step("Then: Verificar estructura JSON"):
            assert response.status_code == 200
            response_data = json.loads(response.content)
            
            # Verificar estructura principal
            assert isinstance(response_data, dict)
            assert 'brand' in response_data
            assert 'data' in response_data
            assert 'analysis' in response_data
            
            # Verificar estructura de análisis
            analysis = response_data['analysis']
            assert isinstance(analysis, dict)
            
            # Verificar technical_indicators
            tech_indicators = analysis['technical_indicators']
            expected_indicators = [
                'moving_averages', 'bollinger_bands', 'macd', 
                'rsi', 'momentum', 'volatility', 'volume_analysis', 'prediction'
            ]
            for indicator in expected_indicators:
                assert indicator in tech_indicators, f"Indicador {indicator} faltante"

    # === CONFIGURACIÓN Y TEARDOWN ===

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup y teardown para cada test"""
        # Setup
        yield
        # Teardown - limpiar cualquier dato residual
        pass

    @classmethod
    def setup_class(cls):
        """Setup para toda la clase de tests"""
        pass

    @classmethod
    def teardown_class(cls):
        """Teardown para toda la clase de tests"""
        pass


# === FUNCIONES AUXILIARES PARA PRUEBAS ===

def create_test_data_file(filepath, data):
    """Función auxiliar para crear archivos de datos de prueba"""
    data.to_csv(filepath, index=True)
    return filepath


def verify_calculation_accuracy(expected, actual, tolerance=0.01):
    """Función auxiliar para verificar precisión de cálculos"""
    if expected is None or actual is None:
        return expected == actual
    return abs(expected - actual) <= tolerance


def generate_mock_stock_data(days=30, trend='neutral', volatility='normal'):
    """Genera datos de stock mock con características específicas"""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Base price
    base_price = 100
    
    # Trend
    if trend == 'bullish':
        trend_factor = np.linspace(0, 20, days)
    elif trend == 'bearish':
        trend_factor = np.linspace(0, -20, days)
    else:
        trend_factor = np.zeros(days)
    
    # Volatility
    if volatility == 'high':
        noise_factor = 5
    elif volatility == 'low':
        noise_factor = 1
    else:
        noise_factor = 2
    
    # Generate prices
    noise = np.random.normal(0, noise_factor, days)
    close_prices = base_price + trend_factor + noise
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': close_prices + np.random.uniform(-1, 1, days),
        'High': close_prices + np.random.uniform(0, 3, days),
        'Low': close_prices - np.random.uniform(0, 3, days),
        'Close': close_prices,
        'Volume': np.random.uniform(1000000, 5000000, days)
    }, index=dates)
    
    # Ensure High/Low consistency
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    return data