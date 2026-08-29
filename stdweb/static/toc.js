/* Table of contents for the task page.
 *
 * Built from the rendered page rather than from a fixed list of steps: which
 * sections a task actually shows depends on which files it has already got,
 * so a list built from anything else would drift and end up pointing at
 * sections that are not there.
 */

// How a step's state is drawn beside its entry. Font Awesome carries the spin
// for us, so a running step needs no animation of its own here.
var TOC_STATES = {
  pending: {icon: 'fa-clock-o',         css: 'text-secondary', title: 'waiting to be run'},
  running: {icon: 'fa-spinner fa-spin', css: 'text-primary',   title: 'running'},
  done:    {icon: 'fa-check',           css: 'text-success',   title: 'done'},
  failed:  {icon: 'fa-times',           css: 'text-danger',    title: 'failed'},
};


function tocApplyState(mark, state) {
  var shown = TOC_STATES[state];

  mark.className = 'toc-state fa ' + (shown ? shown.icon + ' ' + shown.css : '');
  mark.title = shown ? shown.title : '';
}


document.addEventListener('DOMContentLoaded', function() {
  var headings = Array.prototype.slice
      .call(document.querySelectorAll('[data-toc]'))
      .filter(function(heading) { return heading.id; });

  // Whether a chain is going through the steps right now. A task on its first
  // run has only the one section it has got far enough to render, so the count
  // below would hide the rail exactly when its state marks are the point of it.
  var live = headings.some(function(heading) {
    return heading.dataset.tocState === 'running'
        || heading.dataset.tocState === 'pending';
  });

  // Otherwise not worth its own navigation for a couple of sections
  if (headings.length < 3 && !live)
    return;

  var nav = document.createElement('nav');
  nav.id = 'toc';
  nav.className = 'toc';

  var title = document.createElement('div');
  title.className = 'toc-title';
  title.textContent = 'Contents';
  nav.appendChild(title);

  var list = document.createElement('ul');
  list.className = 'nav flex-column';

  headings.forEach(function(heading) {
    var item = document.createElement('li');
    item.className = 'nav-item';

    var link = document.createElement('a');
    link.className = 'nav-link';
    link.href = '#' + heading.id;
    link.title = heading.dataset.tocTitle || heading.textContent.trim();

    // The short name, with the full one left for the tooltip - the rail is
    // deliberately too narrow for 'Transient candidates in difference image'
    var label = document.createElement('span');
    label.className = 'toc-label';
    label.textContent = heading.dataset.toc || heading.textContent.trim();
    link.appendChild(label);

    // How the step stands, which the page keeps up to date as it polls. It
    // lives here rather than beside the heading so that it is in view
    // whichever part of the page is being read.
    var mark = document.createElement('i');
    if (heading.dataset.tocStep)
      mark.dataset.step = heading.dataset.tocStep;
    tocApplyState(mark, heading.dataset.tocState);
    link.appendChild(mark);

    item.appendChild(link);
    list.appendChild(item);
  });

  nav.appendChild(list);
  document.body.appendChild(nav);

  // Where the window is too narrow for the rail to sit beside the content,
  // the same list is reached through a button instead
  var toggle = document.createElement('button');
  toggle.className = 'toc-toggle btn btn-sm btn-outline-secondary';
  toggle.type = 'button';
  toggle.title = 'Contents';
  toggle.innerHTML = '<i class="fa fa-list"></i>';
  document.body.appendChild(toggle);

  toggle.addEventListener('click', function() {
    nav.classList.toggle('toc-open');
  });

  nav.addEventListener('click', function(event) {
    var link = event.target.closest('.nav-link');

    if (!link)
      return;

    // Opened over the page, it should get out of the way once it has been used
    nav.classList.remove('toc-open');

    // Let the browser have the ones meant for a new tab or window
    if (event.button !== 0 || event.metaKey || event.ctrlKey
        || event.shiftKey || event.altKey)
      return;

    var section = document.getElementById(link.getAttribute('href').slice(1));

    if (!section)
      return;

    // Scrolled to rather than navigated to, so the address keeps no record of
    // it: following the link as a link would leave the section in the URL, and
    // every later reload of the page would jump back down to it. The offset
    // that clears the sticky state bar is scroll-margin-top, in the stylesheet.
    event.preventDefault();
    section.scrollIntoView({behavior: 'smooth', block: 'start'});
  });

  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape')
      nav.classList.remove('toc-open');
  });

  // The page hands over each step's state as it polls for it
  window.tocSetStates = function(states) {
    nav.querySelectorAll('.toc-state[data-step]').forEach(function(mark) {
      tocApplyState(mark, states[mark.dataset.step]);
    });
  };

  // Marks whichever section is currently on screen. The bottom margin keeps
  // the highlight on the section being read rather than the one below it.
  new bootstrap.ScrollSpy(document.body, {
    target: '#toc',
    rootMargin: '0px 0px -60%',
  });
});
