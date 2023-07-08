from django import forms

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, Fieldset, Div, Row, Column

from .processing import supported_filters, supported_catalogs


class UploadFileForm(forms.Form):
    file = forms.FileField(label="FITS file")
    title = forms.CharField(max_length=150, required=False, label="Optional title or comment")


class TaskInspectForm(forms.Form):
    form_type = forms.CharField(initial='inspect', widget=forms.HiddenInput())
    target = forms.CharField(max_length=50, required=False, empty_value=None, label="Target name or coordinates")
    gain = forms.FloatField(min_value=0, required=False, label="Gain, e/ADU")
    saturation = forms.FloatField(min_value=0, required=False, label="Saturation level, ADU")
    mask_cosmics = forms.BooleanField(initial=True, required=False, label="Mask cosmics")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            'form_type',
            Row(
                Column('target', css_class="col-md-6"),
                Column('gain', css_class="col-md-3"),
                Column('saturation', css_class="col-md-3"),
                css_class='align-items-end'
            ),
            'mask_cosmics'
        )


class TaskPhotometryForm(forms.Form):
    form_type = forms.CharField(initial='photometry', widget=forms.HiddenInput())
    sn = forms.FloatField(min_value=0, required=False, label="S/N Ratio")
    initial_aper = forms.FloatField(min_value=0, required=False, label="Initial aperture, pixels")
    initial_r0 = forms.FloatField(min_value=0, required=False, label="Smoothing kernel, pixels")
    bg_size = forms.IntegerField(min_value=0, required=False, label="Background mesh size")
    minarea = forms.IntegerField(min_value=0, required=False, label="Minimal object area")
    rel_aper = forms.FloatField(min_value=0, required=False, label="Relative aperture, FWHM")
    rel_bg1 = forms.FloatField(min_value=0, required=False, label="Local background inner annulus, FWHM")
    rel_bg2 = forms.FloatField(min_value=0, required=False, label="Outer annulus, FWHM")

    filter = forms.ChoiceField(choices=[('','')] + [(_,supported_filters[_]['name']) for _ in supported_filters.keys()],
                               required=False, label="Filter")
    cat_name = forms.ChoiceField(choices=[('','')] + [(_,supported_catalogs[_]['name']) for _ in supported_catalogs.keys()],
                                required=False, label="Reference catalog")
    cat_limit = forms.FloatField(required=False, label="Catalog limiting mag")

    spatial_order = forms.IntegerField(min_value=0, required=False, label="Zeropoint spatial order")
    use_color = forms.BooleanField(required=False, label="Use color term")
    refine_wcs = forms.BooleanField(required=False, label="Refine astrometry")
    blind_match_wcs = forms.BooleanField(required=False, label="Blind match")

    blind_match_ps_lo = forms.FloatField(initial=0.2, min_value=0, required=False, label="Scale lower limit, arcsec/pix")
    blind_match_ps_up = forms.FloatField(initial=4.0, min_value=0, required=False, label="Scale upper limit, arcsec/pix")
    blind_match_ra0 = forms.FloatField(min_value=0, max_value=360, required=False, label="Center RA, deg")
    blind_match_dec0 = forms.FloatField(min_value=-90, max_value=90, required=False, label="Center Dec, deg")
    blind_match_sr0 = forms.FloatField(initial=1, min_value=0, required=False, label="Radius, deg")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            'form_type',
            Row(
                Column('sn'),
                Column('initial_aper'),
                Column('initial_r0'),
                Column('bg_size'),
                Column('minarea'),
                css_class='align-items-end'
            ),
            Row(Column('rel_aper'),
                Column('rel_bg1'),
                Column('rel_bg2'),
                Column('spatial_order'),
                css_class='align-items-end'
            ),
            Row(
                Column('filter'),
                Column('cat_name'),
                Column('cat_limit', css_class="col-md-2"),
                Column('use_color', css_class="col-md-2"),
                Column('refine_wcs', css_class="col-md-2"),
                css_class='align-items-end'
            ),
            Row(
                Column('blind_match_wcs', css_class="col-md-2"),
                Column('blind_match_ps_lo', css_class="col-md-2"),
                Column('blind_match_ps_up', css_class="col-md-2"),
                Column('blind_match_ra0', css_class="col-md-2"),
                Column('blind_match_dec0', css_class="col-md-2"),
                Column('blind_match_sr0', css_class="col-md-2"),
                css_class='align-items-end'
                )
        )
