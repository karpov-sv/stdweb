// Interactive, case-insensitive line filtering for FITS header displays.
//
// Any element carrying the `fits-header` class is turned into a filterable
// header. The header markup (see `header_to_string`) already wraps each line in
// a `.fits-header-line` span; a filter input is inserted just above it, and
// typing hides every line that does not contain the entered substring
// (case-insensitively).

(function () {
  'use strict';

  function makeFilterable(el) {
    if (el.dataset.fitsHeaderFilterable) {
      return; // already initialized
    }
    el.dataset.fitsHeaderFilterable = '1';

    // The element carrying the text may be a <code> nested inside a <pre>; put
    // the input above the outer <pre> so it stays within the collapsible card.
    var container = el.closest('pre') || el;

    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'form-control form-control-sm mb-2 fits-header-filter';
    input.placeholder = 'Filter header lines…';
    input.setAttribute('aria-label', 'Filter header lines');
    // Do not let the Enter key submit any surrounding form.
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
      }
    });

    var spans = el.getElementsByClassName('fits-header-line');
    input.addEventListener('input', function () {
      var query = input.value.trim().toLowerCase();
      for (var j = 0; j < spans.length; j++) {
        var span = spans[j];
        var match = !query || span.textContent.toLowerCase().indexOf(query) !== -1;
        span.style.display = match ? '' : 'none';
      }
    });

    container.parentNode.insertBefore(input, container);
  }

  function init(root) {
    var els = (root || document).querySelectorAll('.fits-header');
    for (var i = 0; i < els.length; i++) {
      makeFilterable(els[i]);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { init(); });
  } else {
    init();
  }

  // Expose for content inserted dynamically (e.g. via AJAX partial updates).
  window.initFitsHeaderFilter = init;
})();
