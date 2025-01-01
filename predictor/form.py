from django import forms


class PredictionForm(forms.Form):
    Date = forms.DateField(
        widget=forms.TextInput(attrs={'placeholder': 'dd/mm/yyyy'}),
        input_formats=['%d/%m/%Y'],
        required=True
    )
    Hour = forms.IntegerField(
        min_value=0, max_value=23,
        widget=forms.NumberInput(attrs={'placeholder': '0-23'}),
        required=True
    )
    Holiday = forms.TypedChoiceField(
        choices=[(1, 'Yes'), (0, 'No')],
        widget=forms.Select(),
        coerce=int,
        required=True
    )
    Rain = forms.TypedChoiceField(
        choices=[(1, 'Yes'), (0, 'No')],
        widget=forms.Select(),
        coerce=int,
        required=True
    )
