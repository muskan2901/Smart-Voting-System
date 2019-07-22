from django.conf.urls import url
from django.urls import path
from . import views

app_name = 'Vote'
urlpatterns = [
    path("", views.index, name="index"),
    url(r'^login/', views.user_login, name='login'),
    url(r'^logout/', views.user_logout, name='logout'),
    url(r'^vote/', views.vote, name='vote'),
    url(r'^results/', views.results, name='results'),
    url(r'^home/', views.home, name='home'),
    url(r'^home_hindi/', views.home_hindi, name='home_hindi'),
    url(r'^about/', views.about, name='about'),
    url(r'^voted/', views.voted, name='voted'),
    url(r'^invalid/', views.invalid, name='invalid'),
    url(r'^casted/', views.casted, name='casted'),
    path(r'detect/', views.detect,name='detect'),
    url(r'^create_dataset$', views.create_dataset),
    url(r'^trainer$', views.trainer),
    url(r'^detect$', views.detect),
    url(r'^face_index$', views.face_index, name='face_index'),

]
