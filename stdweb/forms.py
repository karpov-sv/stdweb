from django import forms

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, Div, Row, Column

from .processing import supported_filters, supported_catalogs


class UploadFileForm(forms.Form):
    file = forms.FileField(label="FITS file")
    title = forms.CharField(max_length=150, required=False, label="Optional title or comment")


class TaskInspectForm(forms.Form):
    form_type = forms.CharField(initial='inspect', widget=forms.HiddenInput())
    target = forms.CharField(max_length=50, required=False, empty_value=None, label="Target name or coordinates")
    filter = forms.ChoiceField(choices=[('','')] + [(_,supported_filters[_]['name']) for _ in supported_filters.keys()],
                               required=False, label="Filter")
    gain = forms.FloatField(min_value=0, required=False, label="Gain, e/ADU")
    saturation = forms.FloatField(min_value=0, required=False, label="Saturation level, ADU")
    mask_cosmics = forms.BooleanField(initial=True, required=False, label="Mask cosmics")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.layout = Layout(
            'form_type',
            'target',
            Row(
                Column('filter'),
                Column('gain'),
                Column('saturation'),
                css_class='align-items-end'
            ),
            'mask_cosmics'
        )


class TaskPhotometryForm(forms.Form):
    form_type = forms.CharField(initial='photometry', widget=forms.HiddenInput())
    sn = forms.FloatField(min_value=0, required=False, label="S/N Ratio")
    initial_aper = forms.FloatField(min_value=0, required=False, label="Initial aperture, pixels")
    initial_r0 = forms.FloatField(min_value=0, required=False, label="Smoothing kernel radius, pixels")
    bg_size = forms.IntegerField(min_value=0, required=False, label="Background mesh size")
    minarea = forms.IntegerField(min_value=0, required=False, label="Minimal object area")
    rel_aper = forms.FloatField(min_value=0, required=False, label="Relative aperture, FWHM")

    cat_name = forms.ChoiceField(choices=[('','')] + [(_,supported_catalogs[_]['name']) for _ in supported_catalogs.keys()],
                                required=False, label="Reference catalog")
    cat_limit = forms.FloatField(required=False, label="Catalog limiting mag")

    use_color = forms.BooleanField(required=False, label="Use color term")
    refine_wcs = forms.BooleanField(required=False, label="Refine astrometry")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.layout = Layout(
            'form_type',
            Row(
                Column(Field('sn')),
                Column(Field('initial_aper')),
                Column('initial_r0'),
                Column('rel_aper'),
                Column('bg_size'),
                Column('minarea'),
                css_class='align-items-end'
            ),
            Row(
                Column('cat_name'),
                Column('cat_limit'),
                Column('use_color'),
                Column('refine_wcs'),
                css_class='align-items-end'
            ),
        )
