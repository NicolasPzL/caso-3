from django.test import TestCase

# financialSearch/tests.py

import pytest
import allure
from allure_commons.types import Severity
from django.contrib.auth.models import User
from django.test import Client
from unittest import mock # Para mockear yfinance y analyze_data
import pandas as pd
from datetime import datetime
import numpy as np

# Marcador para que Pytest reconozca las pruebas de Django y maneje la BD
pytestmark = pytest.mark.django_db

# --- Fixtures Comunes ---

@pytest.fixture
@allure.step("Crear un usuario de prueba (clase/clase2024)")
def test_user(db):
    """Fixture para crear un usuario de prueba con username y password."""
    return User.objects.create_user(
        username='clase',
        password='clase2024'
        # email es opcional para create_user si no lo necesitas específicamente
    )

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

# --- Pruebas de Autenticación ---
@allure.feature("Sistema de Autenticación")
class TestAuthentication:
    LOGIN_URL = '/login'
    LOGOUT_URL = '/logout'
    HOME_URL = '/' # Para verificar redirección

    @allure.title("Login Exitoso")
    @allure.description("Verifica que un usuario puede iniciar sesión correctamente con credenciales válidas y es redirigido a la página principal.")
    @allure.tag("auth", "positive", "smoke")
    @allure.severity(Severity.CRITICAL)
    def test_login_successful(self, client, test_user): # Usa el cliente anónimo para el POST de login
        payload = {'username': test_user.username, 'password': 'clase2024'}
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
    @allure.severity(Severity.NORMAL)
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
    @allure.severity(Severity.NORMAL)
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
    @allure.severity(Severity.MINOR)
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
    @allure.severity(Severity.MINOR)
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
    @allure.severity(Severity.CRITICAL)
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
    @allure.severity(Severity.CRITICAL)
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
    @allure.severity(Severity.CRITICAL)
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

    @allure.title("Obtención Exitosa de Datos Financieros")
    @allure.description("Verifica que la vista getReturns procesa una solicitud POST válida, mockeando yfinance y analyze_data, y devuelve el JSON esperado.")
    @allure.tag("returns", "positive", "smoke")
    @allure.severity(Severity.CRITICAL)
    @mock.patch('financialSearch.views.analyze_data') # Mockear tu función de análisis
    @mock.patch('yfinance.Ticker') # Mockear la clase Ticker de yfinance
    def test_get_returns_successful(self, mock_yfinance_ticker_cls, mock_analyze_data_func,
                                     authenticated_client, mock_yfinance_data):
        # 1. Configurar el mock para yfinance.Ticker().history()
        mock_ticker_instance = mock.MagicMock() # Instancia simulada de Ticker
        mock_ticker_instance.history.return_value = mock_yfinance_data # history() devuelve nuestros datos
        mock_yfinance_ticker_cls.return_value = mock_ticker_instance # yf.Ticker(brand) devuelve nuestra instancia mockeada

        # 2. Configurar el mock para analyze_data()
        expected_analysis_result = "Análisis simulado: Recomendación de Compra Fuerte."
        mock_analyze_data_func.return_value = expected_analysis_result

        payload = {
            'from': '2023-01-01',
            'to': '2023-01-05',
            'brand': 'AAPL' # Ticker de ejemplo
        }
        with allure.step(f"POST a {self.GET_RETURNS_URL} con payload: {payload}"):
            response = authenticated_client.post(self.GET_RETURNS_URL, payload)
        
        with allure.step("Verificar respuesta JSON exitosa (200 OK)"):
            assert response.status_code == 200, "La solicitud de getReturns exitosa debería devolver un código 200."
            response_json = response.json()

        with allure.step("Verificar que yfinance.Ticker fue llamado correctamente"):
            mock_yfinance_ticker_cls.assert_called_once_with('AAPL')
        
        with allure.step("Verificar que Ticker.history fue llamado con las fechas correctas"):
            # Convertir strings de fecha del payload a objetos datetime para la aserción
            expected_start_date = datetime.strptime(payload['from'], '%Y-%m-%d')
            expected_end_date = datetime.strptime(payload['to'], '%Y-%m-%d')
            mock_ticker_instance.history.assert_called_once_with(start=expected_start_date, end=expected_end_date)

        with allure.step("Verificar que analyze_data fue llamado con los datos SMA correctos"):
            # Calcular SMA_5 esperado basado en mock_yfinance_data
            expected_sma5_values = mock_yfinance_data['Close'].rolling(window=5).mean().to_list()
            
            # Verificar que el mock fue llamado
            assert mock_analyze_data_func.call_count == 1, "analyze_data debería ser llamado exactamente una vez"
            
            # Obtener los argumentos con los que fue llamado el mock
            actual_args = mock_analyze_data_func.call_args[0][0]
            
            # Verificar que las longitudes coinciden
            assert len(actual_args) == len(expected_sma5_values), "La longitud de los argumentos no coincide"
            
            # Verificar cada valor, manejando nan correctamente
            for actual, expected in zip(actual_args, expected_sma5_values):
                if pd.isna(expected):
                    assert pd.isna(actual), f"Se esperaba nan pero se recibió {actual}"
                else:
                    assert actual == expected, f"Se esperaba {expected} pero se recibió {actual}"
        
        with allure.step("Verificar la estructura y contenido clave del JSON de respuesta"):
            assert response_json['brand'] == 'AAPL', "La marca en la respuesta JSON no coincide con la esperada."
            assert response_json['analysis'] == expected_analysis_result, "El análisis en la respuesta JSON no coincide con el mock."
            assert len(response_json['data']) == len(mock_yfinance_data), "El número de puntos de datos en la respuesta no coincide con los datos mockeados."
            
            # Verificar un punto de datos como ejemplo
            first_data_point_response = response_json['data'][0]
            first_data_point_mock = mock_yfinance_data.iloc[0]
            first_sma_mock = expected_sma5_values[0] # Puede ser NaN si window > 1 y es el primer punto

            assert first_data_point_response['date'] == first_data_point_mock.name.strftime('%Y-%m-%d'), "La fecha del primer punto de datos no coincide."
            assert first_data_point_response['close'] == first_data_point_mock['Close'], "El precio de cierre del primer punto de datos no coincide."
            if pd.isna(first_sma_mock):
                assert first_data_point_response['sma_5'] is None, "El SMA_5 para el primer punto debería ser None si el mock SMA es NaN."
            else:
                assert first_data_point_response['sma_5'] == first_sma_mock, "El SMA_5 para el primer punto de datos no coincide."


    @allure.title("getReturns Requiere Autenticación")
    @allure.description("Verifica que un usuario no autenticado sea redirigido al intentar acceder a getReturns.")
    @allure.tag("returns", "negative", "security")
    @allure.severity(Severity.CRITICAL)
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
    @allure.severity(Severity.NORMAL)
    def test_get_returns_rejects_get_method(self, authenticated_client):
        with allure.step(f"GET a {self.GET_RETURNS_URL} como usuario autenticado"):
            response = authenticated_client.get(self.GET_RETURNS_URL)
        
        with allure.step("Verificar respuesta de error 405 (Método No Permitido)"):
            assert response.status_code == 405, "Una solicitud GET a getReturns debería resultar en un error 405."
            assert response.json()['error'] == 'Método no permitido', "El mensaje de error para método no permitido no es el esperado."

    @allure.title("getReturns Maneja Excepción de yfinance.Ticker().history()")
    @allure.description("Verifica cómo responde la vista si la llamada a yfinance.history() falla.")
    @allure.tag("returns", "negative", "error_handling")
    @allure.severity(Severity.NORMAL)
    @mock.patch('financialSearch.views.analyze_data') # Mockear para aislar
    @mock.patch('yfinance.Ticker')
    def test_get_returns_handles_yfinance_exception(self, mock_yfinance_ticker_cls, mock_analyze_data_func, authenticated_client):
        mock_ticker_instance = mock.MagicMock()
        simulated_error_message = "Error de conexión simulado con yfinance API"
        mock_ticker_instance.history.side_effect = ConnectionError(simulated_error_message) # Simular un error de red
        mock_yfinance_ticker_cls.return_value = mock_ticker_instance
        # analyze_data no debería ser llamado si yfinance falla
        mock_analyze_data_func.return_value = "Este análisis no debería ocurrir"


        payload = {'from': '2023-01-01', 'to': '2023-01-05', 'brand': 'ERROR_TICKER'}
        with allure.step(f"POST a {self.GET_RETURNS_URL} esperando una excepción de yfinance"):
            # Esta prueba ASUME que la vista NO tiene un try-except para la llamada a .history()
            # y que la excepción se propagará, causando un error 500 en Django.
            # Si la vista SÍ maneja la excepción y devuelve un JsonResponse de error,
            # deberás cambiar esta aserción para verificar ese JsonResponse.
            with pytest.raises(ConnectionError, match=simulated_error_message):
                authenticated_client.post(self.GET_RETURNS_URL, payload)
        
        with allure.step("Verificar que analyze_data no fue llamado si yfinance falló"):
            mock_analyze_data_func.assert_not_called()


    @allure.title("getReturns Maneja Formato de Fecha Inválido")
    @allure.description("Verifica que la vista maneje (o falle predeciblemente) formatos de fecha incorrectos.")
    @allure.tag("returns", "negative", "input_validation")
    @allure.severity(Severity.NORMAL)
    def test_get_returns_invalid_date_format(self, authenticated_client):
        payload = {'from': 'fecha_incorrecta_formato', 'to': '2023-01-05', 'brand': 'AAPL'}
        with allure.step(f"POST a {self.GET_RETURNS_URL} con formato de fecha 'from' inválido"):
            # datetime.strptime(from_date, '%Y-%m-%d') en tu vista lanzará ValueError.
            # Asumimos que esta excepción NO es capturada en la vista. Si lo fuera,
            # la prueba debería verificar el JsonResponse de error 400.
            with pytest.raises(ValueError):
                authenticated_client.post(self.GET_RETURNS_URL, payload)

    @allure.title("getReturns Maneja Campos POST Requeridos Faltantes")
    @allure.description("Verifica cómo reacciona la vista si faltan campos esenciales en el payload POST.")
    @allure.tag("returns", "negative", "input_validation")
    @allure.severity(Severity.NORMAL)
    @pytest.mark.parametrize(
        "payload_variant, missing_field_name, expected_error_type, expected_error_message",
        [
            (
                {'to': '2023-01-05', 'brand': 'AAPL'}, 
                "from",
                TypeError,
                "strptime() argument 1 must be str, not None"
            ),
            (
                {'from': '2023-01-01', 'brand': 'AAPL'}, 
                "to",
                TypeError,
                "strptime() argument 1 must be str, not None"
            ),
            (
                {'from': '2023-01-01', 'to': '2023-01-05'}, 
                "brand",
                AttributeError,
                "'NoneType' object has no attribute"
            ),
        ]
    )
    def test_get_returns_missing_required_fields(
        self, 
        authenticated_client, 
        payload_variant, 
        missing_field_name,
        expected_error_type,
        expected_error_message
    ):
        with allure.step(f"POST a {self.GET_RETURNS_URL} faltando el campo '{missing_field_name}'"):
            with pytest.raises(expected_error_type) as exc_info:
                authenticated_client.post(self.GET_RETURNS_URL, payload_variant)
            
            # Verificar el mensaje de error específico de manera más flexible
            error_message = str(exc_info.value)
            
            if missing_field_name in ['from', 'to']:
                # Para campos de fecha, verificar el mensaje de strptime
                assert "strptime() argument 1 must be str" in error_message, \
                    f"El mensaje de error debería mencionar 'strptime() argument 1 must be str', pero fue: '{error_message}'"
                assert "None" in error_message, \
                    f"El mensaje de error debería mencionar 'None', pero fue: '{error_message}'"
            elif missing_field_name == 'brand':
                # Para el campo brand, verificar el mensaje de AttributeError
                assert "NoneType" in error_message, \
                    f"El mensaje de error debería mencionar 'NoneType', pero fue: '{error_message}'"
                assert "has no attribute" in error_message, \
                    f"El mensaje de error debería mencionar 'has no attribute', pero fue: '{error_message}'"
            
            # Verificar que el campo faltante está en el payload
            assert missing_field_name not in payload_variant, \
                f"El campo '{missing_field_name}' no debería estar presente en el payload"
            
            # Verificar que los otros campos están presentes
            for field in ['from', 'to', 'brand']:
                if field != missing_field_name:
                    assert field in payload_variant, \
                        f"El campo '{field}' debería estar presente en el payload"