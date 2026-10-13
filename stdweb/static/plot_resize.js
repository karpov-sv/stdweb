// A plot the reader may make taller or shorter, by dragging the grip along its
// bottom edge.
//
// Plotly draws to whatever size its element has at that moment and does not
// follow it afterwards, so every change of the box holding the plot is handed
// to Plotly.Plots.resize(). That covers the drag, anything else that moves the
// page around it, and the first appearance of a plot drawn while still hidden -
// which had no size at all to be drawn at.

// The shortest a plot may be dragged, pixels
var PLOT_MIN_HEIGHT = 150;

// Makes the plot of that id follow the box it sits in, and its grip drag.
// Neither needs the plot to have been drawn yet, so this may be called as soon
// as the page is ready.
function plotlyResizable(id) {
    var plot = document.getElementById(id);
    var box = plot && plot.closest('.plotly-resizable');
    var handle = box && box.querySelector('.plot-resize-handle');

    if (!handle || box.plotlyResizableBound)
        return;

    box.plotlyResizableBound = true;

    // Once per frame at most, as a drag reports far more moves than there are
    // frames to draw them in
    var pending = null;

    var resize = function() {
        if (pending !== null)
            return;

        pending = window.requestAnimationFrame(function() {
            pending = null;

            // A Plotly element carries its traces once it has been drawn, and
            // nothing before that
            if (plot.data)
                Plotly.Plots.resize(plot);
        });
    };

    new ResizeObserver(resize).observe(box);

    var drag = null;

    handle.addEventListener('pointerdown', function(event) {
        if (event.button !== 0)
            return;

        // The drag is ours, and is not to start a selection of the page
        event.preventDefault();

        drag = {
            pointer: event.pointerId,
            y: event.clientY,
            height: box.getBoundingClientRect().height,
        };

        // So that the pointer keeps reporting to the grip even once it has
        // been dragged off it, which at speed it will be
        handle.setPointerCapture(event.pointerId);
        document.body.classList.add('plot-resizing');
    });

    handle.addEventListener('pointermove', function(event) {
        if (!drag || event.pointerId !== drag.pointer)
            return;

        // Setting the height is all there is to it - the observer above is
        // what tells Plotly about it
        box.style.height = Math.max(PLOT_MIN_HEIGHT,
                                    Math.round(drag.height + event.clientY - drag.y)) + 'px';
    });

    var release = function(event) {
        if (!drag || event.pointerId !== drag.pointer)
            return;

        handle.releasePointerCapture(event.pointerId);
        drag = null;
        document.body.classList.remove('plot-resizing');
    };

    handle.addEventListener('pointerup', release);
    handle.addEventListener('pointercancel', release);
}
