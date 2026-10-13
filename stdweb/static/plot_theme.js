// Plotly colours taken from the page rather than written into every layout,
// so that a plot is in the same theme as everything around it.
//
// Plotly works out the modebar's background, and the shading it dims the page
// with while a zoom box is being dragged, from the two background colours -
// and reads a transparent one as black, alpha and all. So the page's own
// colour is handed over rather than left unset: it changes nothing on screen
// and lets both be worked out correctly in either theme.

function plotlyTheme() {
    var style = getComputedStyle(document.body);

    return {
        paper_bgcolor: style.backgroundColor,
        plot_bgcolor: style.backgroundColor,
        font: {color: style.color},
        // Greys of their own, at opacities that read against a light page and
        // a dark one alike, rather than colours belonging to either
        gridcolor: 'rgba(128,128,128,0.2)',
        // An axis line is drawn over the plot rather than under the data, and
        // wants to be seen a little more than a gridline does
        linecolor: 'rgba(128,128,128,0.5)',
    };
}

// The colours above for a plot that is already drawn. Only what the theme
// decides is touched, so a zoom, a fold or a fitted period all survive it.
function plotlyRetheme(id) {
    var element = document.getElementById(id);

    // A Plotly div carries its traces once it has been drawn, and nothing
    // before that
    if (!element || !element.data)
        return;

    var theme = plotlyTheme();

    Plotly.relayout(element, {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        'font.color': theme.font.color,
        'xaxis.gridcolor': theme.gridcolor,
        'yaxis.gridcolor': theme.gridcolor,
    });
}

// The reader may change theme with the page open - the switch is the system
// one, the stylesheet following prefers-color-scheme
function onThemeChange(callback) {
    var media = window.matchMedia('(prefers-color-scheme: dark)');

    if (media.addEventListener)
        media.addEventListener('change', callback);
    else if (media.addListener)
        media.addListener(callback);   // Safari before 14
}
