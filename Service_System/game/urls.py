from django.urls import path

from . import views

urlpatterns = [
    path("", views.GameListCreateView.as_view(), name="game-list-create"),
    path("<uuid:pk>/", views.GameDetailView.as_view(), name="game-detail"),
    path("worker/claim/", views.worker_claim, name="game-worker-claim"),
    path("worker/<uuid:pk>/image/<str:kind>/", views.worker_image, name="game-worker-image"),
    path("worker/<uuid:pk>/complete/", views.worker_complete, name="game-worker-complete"),
    path("worker/<uuid:pk>/fail/", views.worker_fail, name="game-worker-fail"),
]
