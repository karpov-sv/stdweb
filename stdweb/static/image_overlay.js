$(document).ready(function() {
    /* On loading the page, annotate all images having proper class with the overlay */
    $('.stdview-image').each(function(index) {
        var image = $(this);
        var container = image.wrap("<div/>").parent().addClass('stdview-image-container');

        var overlay = $('<div/>').addClass('stdview-image-overlay');
        container.append(overlay);

        /* Streching and scaling the data, if data-stretch or data-scale parameters are set */
        if ('stretch' in image.data() || 'scale' in image.data()) {
  	    var stretch = $('<select/>');
            var svals = ['linear', 'asinh', 'log'];

            stretch.append($('<option disabled selected>').html('Stretch'));

            for (var i=0; i<svals.length; i++)
    	        stretch.append($('<option/>').html(svals[i]));

            stretch.on('change', function() {update_image_get_params(image, {stretch: this.value})});
            stretch.on('click', function() {return false});
  	    overlay.append(stretch);

            /* Now scaling part */
            var scale = $('<select/>');
            var scvals = [90, 95, 99, 99.5, 99.9, 100];

            scale.append($('<option disabled selected>').html('Scale'));

            for (i=0; i<scvals.length; i++)
    	        scale.append($('<option/>').val(scvals[i]).html(scvals[i].toString()+'%'));

            scale.on('change', function() {update_image_get_params(image, {qmax: this.value})});
            scale.on('click', function() {return false});
  	    overlay.append(scale);
        }

        /* data-mark-ra and data-mark-dec parameters */
        if ('markRa' in image.data() && 'markDec' in image.data()) {
  	    var checkbox = $('<input type="checkbox"/>');
  	    checkbox.on('click', function() {
    	        if (this.checked)
      	            update_image_get_params(image, {ra: image.data('markRa'), dec: image.data('markDec')});
                else
      	            update_image_get_params(image, {ra: null, dec: null});
            });

            checkbox.attr('title', 'Click to mark the position');

  	    overlay.append(checkbox);
        }
    })
});

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
