from django.shortcuts import render, redirect


def index(request):
    return render(request, "landing.html", {})


def feedback(request):
    if request.method == "POST":
        title = request.POST.get("title")
        content = request.POST.get("content")
        return render(request, "feedback.html", {})
    return redirect('dashboard:landing')
