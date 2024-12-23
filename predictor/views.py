from django.shortcuts import render
from django.contrib import messages
from .form import PredictionForm
from .services.predictor import Predictor
from .services.data_handler import DataHandler
from .services.map_visualizer import MapVisualizer


# load the model and data using the loader module
model_name = 'park_pycaret_2012_pipeline'
predictor = Predictor(model_name=model_name)
data_handler = DataHandler()
map_visualizer = MapVisualizer()


def predict(request):
    map_html = None

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            try:
                prediction_input = data_handler.prepare_prediction_data(
                    date=form.cleaned_data['Date'],
                    hour=form.cleaned_data['Hour'],
                    holiday=form.cleaned_data['Holiday'],
                    rain=form.cleaned_data['Rain']
                )

                predictions = predictor.predict(prediction_input)
                map_data = data_handler.prepare_map_data(predictions)

                map_visualizer.add_markers(map_data)
                map_html = map_visualizer.render_map()

                messages.success(request, 'Prediction completed.')
            except ValueError as e:
                messages.error(request, f"Error: {str(e)}")
        else:
            messages.error(request, 'Invalid form submission, please try again.')
    else:
        form = PredictionForm()

    return render(request, 'predictor/index.html', {
        'form': form,
        'map_html': map_html
    })
