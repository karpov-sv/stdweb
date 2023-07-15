$(document).ready(function(){
    // Initialize popovers from spans
    $('.span-help-popover').each(function(index) {
        var x=$(this);
        x.addClass("fa fa-question-circle small");
        x.addClass("text-info");
        x.attr('data-bs-trigger', 'hover focus');
        // x.attr('data-bs-placement', 'left');
        x.attr('data-bs-custom-class', 'help-popover');
        x.attr('data-bs-content', x.html());
        x.html(''); // Remove the text from span itself, as it is in the popover now.

        const popover = new bootstrap.Popover(x, {html: true, container: 'body'});
    });
});
