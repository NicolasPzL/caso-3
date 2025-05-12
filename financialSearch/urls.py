from django.urls import path
from . import views

app_name = 'financialSearch'

urlpatterns = [
    path('login', views.user_login, name='login'),
    path('logout', views.user_logout, name='logout'),
    path('getReturns', views.getReturns, name='getReturns'),
    path('', views.home, name='home'),
]