from django.urls import path, include
from . import views

app_name = "main_page"
urlpatterns = [
    path('', views.read, name='read'),
    path('link/', views.scrape, name='scrape'),
    path("check-domain/", views.check_domain, name="check_domain"),
    path("stream/", views.stream_process, name="stream_process"),
    # path('create/', views.create, name='create'),
    # path('update/<slug:id_dosen>/', views.update, name='update'),
    # path('delete/<slug:id_dosen>/', views.delete, name='delete'),
    # path('check/', views.permission, name='permission'),
    # path('<slug:id_dosen>/', views.readDetail, name='read-detail'),
]
