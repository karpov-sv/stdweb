from django import forms

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, Fieldset, Div, Row, Column, Submit
from crispy_forms.bootstrap import InlineField, PrependedText, InlineRadios

# from django_select2 import forms as s2forms

import json

from .processing import supported_filters, supported_catalogs, supported_catalogs_transients, supported_templates


class UploadFileForm(forms.Form):
    file = forms.FileField(label="FITS file")
    title = forms.CharField(max_length=150, required=False, label="Optional title or comment")


class TasksFilterForm(forms.Form):
    query = forms.CharField(max_length=100, required=False, label="Filter Tasks")
    show_all = forms.BooleanField(initial=False, required=False, label="Show all")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'GET'
        self.helper.form_action = 'tasks'
        self.helper.form_show_labels = False
        self.helper.layout = Layout(
            Row(
                Column(
                    InlineField(
                        PrependedText('query', 'Filter:', placeholder='Search tasks by filenames or titles or usernames'),
                    ),
                    css_class="col-md"
                ),
                Column(
                    InlineField('show_all'),
                    css_class="col-md-auto"
                ),
            )
        )


class PrettyJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, indent, sort_keys, **kwargs):
        super().__init__(*args, indent=4, sort_keys=False, **kwargs)


class TaskInspectForm(forms.Form):
    form_type = forms.CharField(initial='inspect', widget=forms.HiddenInput())
    target = forms.CharField(required=False, empty_value=None, label="Target name or coordinates",
                             widget=forms.Textarea(attrs={'rows':1, 'placeholder': 'Name or coordinates, one per line'}))
    time = forms.CharField(max_length=30, required=False, empty_value=None, label="Time")
    gain = forms.FloatField(min_value=0, required=False, label="Gain, e/ADU")
    saturation = forms.FloatField(min_value=0, required=False, label="Saturation level, ADU")
    mask_cosmics = forms.BooleanField(initial=True, required=False, label="Mask cosmics")

    raw_config = forms.JSONField(initial=False, required=False, label="Raw config JSON", encoder=PrettyJSONEncoder)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            'form_type',
            Row(
                Column('target', css_class="col-md-5"),
                Column('time', css_class="col-md-3"),
                Column('gain', css_class="col-md-2"),
                Column('saturation', css_class="col-md-2"),
                css_class='align-items-end'
            ),
            Row(
                Column('mask_cosmics', css_class="col-md-auto"),
                css_class='align-items-end justify-content-start'
            ),
        )


class TaskPhotometryForm(forms.Form):
    form_type = forms.CharField(initial='photometry', widget=forms.HiddenInput())
    sn = forms.FloatField(min_value=0, required=False, label="S/N Ratio")
    initial_aper = forms.FloatField(min_value=0, required=False, label="Initial aperture, pixels")
    initial_r0 = forms.FloatField(min_value=0, required=False, label="Smoothing kernel, pixels")
    bg_size = forms.IntegerField(min_value=0, required=False, label="Background mesh size")
    minarea = forms.IntegerField(min_value=0, required=False, label="Minimal object area")
    rel_aper = forms.FloatField(min_value=0, required=False, label="Relative aperture, FWHM")
    rel_bg1 = forms.FloatField(min_value=0, required=False, label="Sky inner annulus, FWHM")
    rel_bg2 = forms.FloatField(min_value=0, required=False, label="Outer annulus, FWHM")
    fwhm_override = forms.FloatField(min_value=0, required=False, label="FWHM override, pixels")

    filter = forms.ChoiceField(choices=[('','')] + [(_,supported_filters[_]['name']) for _ in supported_filters.keys()],
                               required=False, label="Filter")
    cat_name = forms.ChoiceField(choices=[('','')] + [(_,supported_catalogs[_]['name']) for _ in supported_catalogs.keys()],
                                required=False, label="Reference catalog")
    cat_limit = forms.FloatField(required=False, label="Catalog limiting mag")

    spatial_order = forms.IntegerField(min_value=0, required=False, label="Zeropoint spatial order")
    use_color = forms.BooleanField(required=False, label="Use color term")
    sr_override = forms.FloatField(min_value=0, required=False, label="Matching radius, arcsec")

    prefilter_detections = forms.BooleanField(initial=True, required=False, label="Pre-filter detections")
    diagnose_color = forms.BooleanField(initial=False, required=False, label="Color term diagnostics")
    refine_wcs = forms.BooleanField(required=False, label="Refine astrometry")
    blind_match_wcs = forms.BooleanField(required=False, label="Blind match")
    inspect_bg = forms.BooleanField(required=False, label="Inspect background")

    blind_match_ps_lo = forms.FloatField(initial=0.2, min_value=0, required=False, label="Scale lower limit, arcsec/pix")
    blind_match_ps_up = forms.FloatField(initial=4.0, min_value=0, required=False, label="Scale upper limit, arcsec/pix")
    blind_match_center = forms.CharField(required=False, empty_value=None, label="Center position for blind match")
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
                Column('fwhm_override'),
                Column('sr_override'),
                css_class='align-items-end'
            ),
            Row(
                Column('filter'),
                Column('cat_name'),
                Column('cat_limit', css_class="col-md-2"),
                Column('spatial_order', css_class="col-md-2"),
                Column('use_color', css_class="col-md-2"),
                css_class='align-items-end'
            ),
            Row(
                Column('blind_match_ps_lo', css_class="col-md-3"),
                Column('blind_match_ps_up', css_class="col-md-3"),
                Column('blind_match_center', css_class="col-md-4"),
                Column('blind_match_sr0', css_class="col-md-2"),
                id='blind_match_params_row',
                css_class='align-items-end'
            ),
            Row(
                Column('refine_wcs', css_class="col-md-auto"),
                Column('blind_match_wcs', css_class="col-md-auto"),
                Column('prefilter_detections', css_class="col-md-auto"),
                Column('diagnose_color', css_class="col-md-auto"),
                Column('inspect_bg', css_class="col-md-auto"),
                css_class='align-items-end'
            ),
        )


class TaskTransientsSimpleForm(forms.Form):
    form_type = forms.CharField(initial='transients_simple', widget=forms.HiddenInput())
    # simple_vizier = forms.MultipleChoiceField(
    #     initial=['ps1', 'skymapper'],
    #     choices=[(_,supported_catalogs_transients[_]['name']) for _ in supported_catalogs_transients.keys()],
    #     required=False,
    #     label="Vizier catalogues",
    #     widget=s2forms.Select2MultipleWidget,
    # )
    simple_skybot = forms.BooleanField(initial=True, required=False, label="SkyBoT")
    simple_others = forms.CharField(initial=None, empty_value=None, required=False, label="Task IDs to cross-check")
    simple_center = forms.CharField(required=False, empty_value=None, label="Center position to limit the search")
    simple_sr0 = forms.FloatField(initial=None, min_value=0, required=False, label="Radius, deg")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            'form_type',
            Row(
                # Column('simple_vizier', css_class="col-md-4"),
                Column('simple_others', css_class="col-md-6"),
                Column('simple_skybot', css_class="col-md-1"),
                Column('simple_center', css_class="col-md-4"),
                Column('simple_sr0', css_class="col-md-1"),
                css_class='align-items-end'
            ),
        )


class TaskSubtractionForm(forms.Form):
    form_type = forms.CharField(initial='subtraction', widget=forms.HiddenInput())
    template = forms.ChoiceField(choices=[(_,supported_templates[_]['name']) for _ in supported_templates.keys()],
                                 required=False, label="Template")
    hotpants_extra = forms.JSONField(required=False, label="HOTPANTS extra params", widget=forms.TextInput)
    sub_size = forms.IntegerField(min_value=0, required=False, label="Sub-image size")
    sub_overlap = forms.IntegerField(min_value=0, required=False, label="Sub-image overlap")
    sub_verbose = forms.BooleanField(required=False, label="Verbose")
    custom_template = forms.FileField(required=False, label="Custom template file")
    custom_template_gain = forms.FloatField(min_value=0, required=False, label="Custom template gain, e/ADU")
    custom_template_saturation = forms.FloatField(min_value=0, required=False, label="Saturation level, ADU")

    subtraction_mode = forms.ChoiceField(choices=[('target', 'Target photometry'), ('detection', 'Transient detection')],
                                         initial='detection', required=True, label="", widget=forms.RadioSelect)

    subtraction_method = forms.ChoiceField(choices=[('zogy', 'ZOGY'), ('hotpants', 'HOTPANTS')],
                                         initial='hotpants', required=False, label="Method")

    filter_vizier = forms.BooleanField(initial=False, required=False, label="Filter Vizier catalogues")
    filter_skybot = forms.BooleanField(initial=False, required=False, label="Filter SkyBoT")
    filter_adjust = forms.BooleanField(initial=True, required=False, label="Sub-pixel adjustment")
    filter_center = forms.CharField(required=False, empty_value=None, label="Center position to limit the search")
    filter_sr0 = forms.FloatField(initial=1, min_value=0, required=False, label="Radius, deg")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            'form_type',
            Row(
                Column('template'),
                # Column('file'),
                Column('sub_size', css_class="col-md-2"),
                Column('sub_overlap', css_class="col-md-2"),
                Column('subtraction_method', css_class="col-md-2"),
                Column('hotpants_extra', id='hotpants_extra_col'),
                css_class='align-items-end'
            ),
            Row(
                Column('custom_template'),
                Column('custom_template_gain'),
                Column('custom_template_saturation'),
                css_class='align-items-end',
                id='custom_row',
            ),
            Row(
                Column('filter_center', css_class="col-md-4"),
                Column('filter_sr0', css_class="col-md-1"),
                Column('filter_vizier', css_class="col-md-auto"),
                Column('filter_skybot', css_class="col-md-auto"),
                Column('filter_adjust', css_class="col-md-auto"),
                css_class='align-items-end',
                id='transients_row',
            ),
            Row(
                Column(InlineRadios('subtraction_mode', template='crispy_radioselect_inline.html'), css_class='form-group'),
                Div(css_class="col-md"),
                Column('sub_verbose', css_class="col-md-1"),
                css_class='align-items-end'
            ),
        )


class SkyPortalUploadForm(forms.Form):
    ids = forms.CharField(max_length=150, required=False, label="Task IDs to upload")
    instrument = forms.ChoiceField(choices=[], initial=None, required=False, label="Instrument")

    def __init__(self, *args, **kwargs):
        instruments = kwargs.pop('instruments')
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.form_show_labels = False
        self.helper.layout = Layout(
            Row(
                Column(
                    InlineField(
                        PrependedText('ids', 'IDs:', placeholder='Comma or whitespace separated list of task IDs to upload'),
                    ),
                    css_class="col-md"
                ),
                Column(
                    'instrument',
                    css_class="col-md-auto"
                ),
                Column(
                    Submit('preview', 'Preview', css_class='btn-primary'),
                    css_class="col-md-auto"
                )
            )
        )

        if instruments is not None:
            self.fields['instrument'].choices = instruments
