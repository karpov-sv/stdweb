$(document).ready(function() {
    /* On loading the page, annotate all images having proper class with the overlay */
    overlay_stdview_images();
});

overlay_stdview_images = function() {
    $('.stdview-image').each(function(index) {
        var image = $(this);
        var container = image.wrap("<div/>").parent().addClass('stdview-image-container');

        var overlay = $('<div/>').addClass('stdview-image-overlay');
        container.append(overlay);

        /* Streching and scaling the data, if data-stretch or data-scale parameters are set */
        if ('stretch' in image.data() || 'scale' in image.data()) {
  	    var stretch = $('<select/>');
            var svals = ['linear', 'asinh', 'log', 'sqrt', 'sinh', 'power', 'histeq'];

            stretch.append($('<option disabled selected>').html('Stretch'));

            for (var i=0; i<svals.length; i++)
    	        stretch.append($('<option/>').html(svals[i]));

            stretch.on('change', function() {update_image_get_params(image, {stretch: this.value})});
            stretch.on('click', function() {return false});
  	    overlay.append(stretch);

            /* Now scaling part */
            var scale = $('<select/>');
            var scvals = [90, 95, 99, 99.5, 99.9, 99.995, 100];

            scale.append($('<option disabled selected>').html('Scale'));

            for (i=0; i<scvals.length; i++)
    	        scale.append($('<option/>').val(scvals[i]).html(scvals[i].toString()+'%'));

            scale.on('change', function() {update_image_get_params(image, {qmax: this.value})});
            scale.on('click', function() {return false});
  	    overlay.append(scale);
        }

        /* Zoom/pan */
        if ('zoom' in image.data()) {
  	    var zoom = $('<select/>');
            var zvals = [1, 2, 4, 8, 16, 32];

            zoom.append($('<option disabled selected>').html('Zoom'));

            for (i=0; i<zvals.length; i++)
    	        zoom.append($('<option/>').val(zvals[i]).html('x'+zvals[i].toString()));

            zoom.on('change', function() {
                if (this.value == 1) {
                    update_image_get_params(image, {zoom: this.value, dx: null, dy: null});
                } else {
                    update_image_get_params(image, {zoom: this.value});
                }
            });
            zoom.on('click', function() {return false});

            image.on('click', function(evt) {update_image_pos(image, evt)});

  	    overlay.append(zoom);

        }

        /* Smooth */
        if ('smooth' in image.data()) {
  	    var smooth = $('<select/>');
            var svals = [0, 0.5, 1, 2, 4];

            smooth.append($('<option disabled selected>').html('Smooth'));

            for (i=0; i<svals.length; i++)
    	        smooth.append($('<option/>').val(svals[i]).html(svals[i].toString()+' px'));

            smooth.on('change', function() {
                if (this.value == 0) {
                    update_image_get_params(image, {r0: null});
                } else {
                    update_image_get_params(image, {r0: this.value});
                }
            });
            smooth.on('click', function() {return false});

  	    overlay.append(smooth);

        }

        /* data-mark-ra and data-mark-dec parameters */
        if ('markRa' in image.data() && 'markDec' in image.data()) {
  	    var checkbox = $('<input type="checkbox"/>');
            var label = $('<i class="fa fa-bullseye" style="padding-left: 0.3em; padding-right: 0.1em;">');

  	    checkbox.on('click', function() {
    	        if (this.checked) {
      	            update_image_get_params(image, {ra: image.data('markRa'), dec: image.data('markDec')});

                    if ('markRadius' in image.data())
                        update_image_get_params(image, {radius: image.data('markRadius')});
                    if ('markRadius2' in image.data())
                        update_image_get_params(image, {radius2: image.data('markRadius2')});
                    if ('markRadius3' in image.data())
                        update_image_get_params(image, {radius3: image.data('markRadius3')});
                } else
      	            update_image_get_params(image, {ra: null, dec: null, r0: null});
            });

            checkbox.attr('title', 'Click to mark the position');

  	    overlay.append(label);
  	    overlay.append(checkbox);
        }

        /* grid */
        if ('grid' in image.data()) {
  	    var checkbox = $('<input type="checkbox" id="checkbox_grid"/>');
            var label = $('<i class="fa fa-th" style="padding-left: 0.3em;  padding-right: 0.1em;">');

  	    checkbox.on('click', function() {
    	        if (this.checked)
      	            update_image_get_params(image, {grid: 1});
                else
      	            update_image_get_params(image, {grid: null});
            });

            checkbox.attr('title', 'Click to show grid');

  	    overlay.append(label);
  	    overlay.append(checkbox);

        }

        /* objects */
        if ('obj' in image.data()) {
  	    var checkbox = $('<input type="checkbox" id="checkbox_obj"/>');
            var label = $('<i class="fa fa-star-half-o" style="padding-left: 0.3em;  padding-right: 0.1em;">');

  	    checkbox.on('click', function() {
    	        if (this.checked)
      	            update_image_get_params(image, {obj: 1});
                else
      	            update_image_get_params(image, {obj: null});
            });

            checkbox.attr('title', 'Click to show detected objects');

  	    overlay.append(label);
  	    overlay.append(checkbox);

        }

        /* objects */
        if ('cat' in image.data()) {
  	    var checkbox = $('<input type="checkbox" id="checkbox_cat"/>');
            var label = $('<i class="fa fa-star-o" style="padding-left: 0.3em;  padding-right: 0.1em;">');

  	    checkbox.on('click', function() {
    	        if (this.checked)
      	            update_image_get_params(image, {cat: 1});
                else
      	            update_image_get_params(image, {cat: null});
            });

            checkbox.attr('title', 'Click to show catalogue stars');

  	    overlay.append(label);
  	    overlay.append(checkbox);

        }
    });
}

update_image_get_params = function(id, params) {
    /* Show 'loading' overlay */
    $(id).parent().addClass('stdview-image-loading');

    $(id).one('load', function() {
        /* Hide 'loading' overlay */
        $(id).parent().removeClass('stdview-image-loading');
    }).attr('src', function(i, src) {
        var url = new URL(src, window.location);

        for (var name in params) {
            if (params[name] === null)
                url.searchParams.delete(name);
            else
                url.searchParams.set(name, params[name]);
        }

        return url.href;
    });
}

update_image_pos = function(id, evt) {
    var url = new URL($(id).attr('src'), window.location);

    var zoom = Number(url.searchParams.get('zoom'));

    if (zoom > 1) {
        var x = evt.offsetX / $(id).width();
        var y = evt.offsetY / $(id).height();

        var dx = (x > 0.75) ? 1/zoom : ((x < 0.25) ? -1/zoom : 0);
        var dy = (y < 0.25) ? 1/zoom : ((y > 0.75) ? -1/zoom : 0);

        if (dx != 0 || dy != 0) {
            dx += Number(url.searchParams.get('dx'));
            dy += Number(url.searchParams.get('dy'));

            update_image_get_params(id, {dx: dx, dy: dy});
        }
    }
}
