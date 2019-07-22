# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from .models import Records
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def details(request, id):
    record = Records.objects.get(id=id)
    context = {
        'record' : record
    }
    return render(request, 'details.html', context)
