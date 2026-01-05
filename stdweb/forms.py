from django import forms

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, Fieldset, Div, Row, Column, Submit, HTML
from crispy_forms.bootstrap import InlineField, PrependedText, InlineRadios

# from django_select2 import forms as s2forms

import json

from .processing import supported_filters, supported_catalogs, supported_catalogs_transients, supported_templates
from . import models


class MultipleChoiceFieldNoValidation(forms.MultipleChoiceField):
    def validate(self, value):
        pass


class UploadFileForm(forms.Form):
    file = forms.FileField(label="FITS file", required=False)
    local_file = forms.CharField(required=False, widget=forms.HiddenInput())
    local_filename = forms.CharField(required=False, label="FITS file", disabled=True) # Will not be sent
    local_files = MultipleChoiceFieldNoValidation(required=False, widget=forms.CheckboxSelectMultiple)
    preset = forms.ChoiceField(
        choices=[('','')],
        required=False, label="Configuration Preset"
    )
    target = forms.CharField(
        required=False, empty_value=None, label="Target name or coordinates",
        widget=forms.Textarea(attrs={'rows':1, 'placeholder': 'Name or coordinates, one per line'})
    )

    do_inspect = forms.BooleanField(initial=False, required=False, label="Inspection")
    do_photometry = forms.BooleanField(initial=False, required=False, label="Photometry")
    do_simple_transients = forms.BooleanField(initial=False, required=False, label="Simple transients detection")
    do_subtraction = forms.BooleanField(initial=False, required=False, label="Subtraction")

    title = forms.CharField(max_length=150, required=False, label="Optional title or comment")

    # FIXME: changes to these fields should be reflected in views.upload_file() !!!
    stack_method = forms.ChoiceField(
        choices=[
            ('sum', 'Sum'),
            ('clipped_mean', 'Sigma-clipped mean'),
            ('median', 'Median'),
        ],
        required=False, label="Stacking method"
    )
    stack_subtract_bg = forms.BooleanField(initial=True, required=False, label="Subtract background")
    stack_mask_cosmics = forms.BooleanField(initial=False, required=False, label="Mask cosmics")

    def __init__(self, *args, **kwargs):
        filename = kwargs.pop('filename', None)
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.form_action = 'upload'
        self.helper.field_template = 'crispy_field.html'

        if filename == '*':
            file_field = 'local_filename'
            self.fields['local_file'].initial = filename
            self.fields['local_filename'].initial = 'Please select one or more files above'
            submit = Submit('process_files', 'Process selected files', css_class='btn-primary')
        elif filename:
            file_field = 'local_filename'
            self.fields['local_file'].initial = filename
            self.fields['local_filename'].initial = filename
            submit = Submit('process', 'Process this file', css_class='btn-primary')
        else:
            file_field = 'file'
            self.fields['file'].required = True
            submit = Submit('upload', 'Upload', css_class='btn-primary')

        self.helper.layout = Layout(
            Row(
                Column(file_field, css_class="col-md"),
                'local_file',
                Column('preset', css_class="col-md-auto"),
                Column(submit, css_class="col-md-auto mb-1"),
                Column(
                    Submit('stack_files', 'Stack and Process', css_class='btn-secondary'),
                    css_class="col-md-auto mb-1"
                ) if filename == '*' else None,
                css_class='align-items-end'
            ),
            Row(
                Column('title', css_class="col-md"),
                Column('target', css_class="col-md"),
                css_class='align-items-end'
            ),
            Row(
                Column('stack_method', css_class="col-md"),
                Column('stack_subtract_bg', css_class="col-md-auto mb-2"),
                Column('stack_mask_cosmics', css_class="col-md-auto mb-2"),
                css_class='align-items-end'
            ) if filename == '*' else None,
            Row(
                Column(HTML("Run automatically:"), css_class="col-md-auto mb-1"),
                Column('do_inspect', css_class="col-md-auto"),
                Column('do_photometry', css_class="col-md-auto"),
                Column('do_simple_transients', css_class="col-md-auto"),
                Column('do_subtraction', css_class="col-md-auto"),
                css_class='align-items-end justify-content-start'
            ),
        )

        # Populate presets
        self.fields['preset'].choices = [('','')] + [(_.id, _.name) for _ in models.Preset.objects.all()]


class TasksFilterForm(forms.Form):
    query = forms.CharField(max_length=100, required=False, label="Filter Tasks")
    show_all = forms.BooleanField(initial=False, required=False, label="Show all")

    def __init__(self, *args, show_all=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'GET'
        self.helper.form_action = 'tasks'
        self.helper.form_show_labels = False
        self.helper.layout = Layout(
            Row(
                Column(
                    InlineField(
                        PrependedText('query', 'Filter:', placeholder='Search tasks by filenames or titles or usernames, or specify field center (and optionally radius) for positional search.'),
                    ),
                    css_class="col-md"
                ),
                Column(
                    InlineField('show_all'),
                    css_class="col-md-auto mt-2"
                ) if show_all else None,
            )
        )


class TasksActionsForm(forms.Form):
    tasks = MultipleChoiceFieldNoValidation(required=False, widget=forms.CheckboxSelectMultiple)
    referer = forms.CharField(widget=forms.HiddenInput())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.form_action = 'tasks_actions'
        self.helper.layout = Layout()


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

    run_photometry = forms.BooleanField(initial=False, required=False, label="Photometry")
    run_subtraction = forms.BooleanField(initial=False, required=False, label="Subtraction")

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
                Column(
                    Row(
                        Column(HTML('Also run:'), css_class="col-md-auto"),
                        Column('run_photometry', css_class="col-md-auto"),
                        Column('run_subtraction', css_class="col-md-auto"),
                        css_class='align-items-start justify-content-start'
                    ),
                    css_class="col-md-auto"
                ),
                css_class='align-items-end justify-content-between'
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
    filter_blends = forms.BooleanField(initial=True, required=False, label="Filter catalogue blends")
    diagnose_color = forms.BooleanField(initial=False, required=False, label="Color term diagnostics")
    refine_wcs = forms.BooleanField(required=False, label="Refine astrometry")
    blind_match_wcs = forms.BooleanField(required=False, label="Blind match")
    inspect_bg = forms.BooleanField(required=False, label="Inspect background")
    centroid_targets = forms.BooleanField(required=False, label="Centroid targets")
    nonlin = forms.BooleanField(required=False, label="Non-linearity")

    blind_match_ps_lo = forms.FloatField(initial=0.2, min_value=0, required=False, label="Scale lower limit, arcsec/pix")
    blind_match_ps_up = forms.FloatField(initial=4.0, min_value=0, required=False, label="Scale upper limit, arcsec/pix")
    blind_match_center = forms.CharField(required=False, empty_value=None, label="Center position for blind match")
    blind_match_sr0 = forms.FloatField(initial=2, min_value=0, required=False, label="Radius, deg")

    run_subtraction = forms.BooleanField(initial=False, required=False, label="Subtraction")

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
            Row(
                Column('rel_aper'),
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
                Column('use_color', css_class="col-md-2 mb-2"),
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
                Column('filter_blends', css_class="col-md-auto"),
                Column('prefilter_detections', css_class="col-md-auto"),
                Column('centroid_targets', css_class="col-md-auto"),
                Column('nonlin', css_class="col-md-auto"),
                Column('diagnose_color', css_class="col-md-auto"),
                Column('inspect_bg', css_class="col-md-auto"),
                css_class='align-items-end'
            ),
            Row(
                Column(HTML('Also run:'), css_class="col-md-auto"),
                # Column('run_photometry', css_class="col-md-auto"),
                Column('run_subtraction', css_class="col-md-auto"),
                css_class='align-items-start justify-content-end'
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
    simple_skybot = forms.BooleanField(initial=True, required=False, label="Check SkyBoT")
    simple_others = forms.CharField(initial=None, empty_value=None, required=False, label="Task IDs to cross-check")
    simple_center = forms.CharField(required=False, empty_value=None, label="Center position to limit the search")
    simple_sr0 = forms.FloatField(initial=None, min_value=0, required=False, label="Radius, deg")
    simple_blends = forms.BooleanField(initial=True, required=False, label="Reject blends")
    simple_prefilter = forms.BooleanField(initial=True, required=False, label="Reject prefiltered")
    simple_mag_diff = forms.FloatField(initial=2, min_value=0, required=False, label="Minimal mag difference")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            'form_type',
            Row(
                Column('simple_center', css_class="col-md-4"),
                Column('simple_sr0', css_class="col-md-1"),
                Column('simple_mag_diff', css_class="col-md-2"),
                # Column('simple_vizier', css_class="col-md-4"),
                Column('simple_others', css_class="col-md-5"),
                css_class='align-items-end'
            ),
            Row(
                Column('simple_skybot', css_class="col-md-auto"),
                Column('simple_blends', css_class="col-md-auto"),
                Column('simple_prefilter', css_class="col-md-auto"),
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
    template_fwhm_override = forms.FloatField(min_value=0, required=False, label="Template FWHM override, image pixels")
    custom_template_gain = forms.FloatField(min_value=0, required=False, label="Custom template gain, e/ADU")
    custom_template_saturation = forms.FloatField(min_value=0, required=False, label="Saturation level, ADU")

    subtraction_mode = forms.ChoiceField(choices=[('target', 'Target photometry'), ('detection', 'Transient detection')],
                                         initial='detection', required=True, label="", widget=forms.RadioSelect)

    subtraction_method = forms.ChoiceField(choices=[('zogy', 'ZOGY'), ('hotpants', 'HOTPANTS')],
                                         initial='hotpants', required=False, label="Method")

    filter_vizier = forms.BooleanField(initial=False, required=False, label="Filter Vizier catalogues")
    filter_skybot = forms.BooleanField(initial=False, required=False, label="Filter SkyBoT")
    filter_prefilter = forms.BooleanField(initial=True, required=False, label="Pre-filtering")
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
                Column('subtraction_method', css_class="col-md-2 d-none"),
                Column('template_fwhm_override'),
                Column('hotpants_extra', id='hotpants_extra_col'),
                css_class='align-items-end'
            ),
            Row(
                Column('custom_template'),
                Column('custom_template_gain', css_class="col-md-2"),
                Column('custom_template_saturation', css_class="col-md-2"),
                Column(
                    Submit('action_custom_mask', 'Make template mask', css_class='btn-secondary mb-1'),
                    css_class="col-md-auto"
                ),
                css_class='align-items-end',
                id='custom_row',
            ),
            Row(
                Column('filter_center', css_class="col-md-4"),
                Column('filter_sr0', css_class="col-md-1"),
                Column('filter_vizier', css_class="col-md-auto mb-2"),
                Column('filter_skybot', css_class="col-md-auto mb-2"),
                Column('filter_prefilter', css_class="col-md-auto mb-2"),
                Column('filter_adjust', css_class="col-md-auto mb-2"),
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
    ids = forms.CharField(
        max_length=150, required=False, label="Task IDs to upload",
        widget=forms.TextInput(attrs={'placeholder': 'Comma or whitespace separated list of task IDs to upload'})
    )
    types = forms.ChoiceField(choices=[
        ('best', 'Best'), ('direct', 'Direct'), ('subtracted', 'Template-subtracted'),
    ],  initial='best', required=False, label="Photometry type")
    instrument = forms.ChoiceField(choices=[], initial=None, required=False, label="Instrument")
    limit_only = forms.BooleanField(initial=False, required=False, label="Upload upper limit only")

    def __init__(self, *args, **kwargs):
        instruments = kwargs.pop('instruments')
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.disable_csrf = True
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            Row(
                Column(
                    'ids',
                    css_class="col-md"
                ),
                Column(
                    'types',
                    css_class="col-md-auto"
                ),
                Column(
                    'instrument',
                    css_class="col-md-auto"
                ),
                Column(
                    Submit('preview', 'Preview', css_class='btn-primary mb-1'),
                    css_class="col-md-auto"
                ),
                css_class='align-items-end',
            ),
            Row(
                Column('limit_only', css_class="col-md"),
                css_class='align-items-end',
            ),
        )

        if instruments is not None:
            self.fields['instrument'].choices = instruments


class LightcurveSearchForm(forms.Form):
    coordinates = forms.CharField(
        max_length=200,
        required=True,
        label="Sky Position",
        widget=forms.TextInput(attrs={'placeholder': 'Name or coordinates'}),
    )
    radius = forms.FloatField(min_value=0, required=False, label="Search radius, arcsec")
    show_images = forms.BooleanField(initial=True, required=False, label="Show images")
    targets_only = forms.BooleanField(initial=True, required=False, label="Target photometry only")
    show_all = forms.BooleanField(initial=True, required=False, label="Tasks from all users")

    def __init__(self, *args, show_all=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'GET'
        self.helper.form_action = 'lightcurves'
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            Row(
                Column('coordinates', css_class="col-md"),
                Column('radius', css_class="col-md-auto"),
                Column(Submit('search', 'Search', css_class='btn-primary mb-1'), css_class="col-md-auto"),
                css_class='align-items-end',
            ),
            Row(
                Column('show_images', css_class="col-md-auto"),
                Column('targets_only', css_class="col-md-auto"),
                Column('show_all', css_class="col-md-auto") if show_all else None,
                css_class='align-items-end',
            )
        )
