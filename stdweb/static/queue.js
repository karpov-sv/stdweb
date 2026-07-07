// Interactive behaviour for the Celery queue pages: auto-refreshing the queue
// list, live task state polling, and AJAX action buttons (terminate / cleanup)

$(function(){
  var refresh_timeout = 3000;

  // --- Queue list auto-refresh ---
  var list = $('#queue-list-container');
  var list_timer = null;

  function scheduleListRefresh(){
    clearTimeout(list_timer);
    if (!document.hidden)
      list_timer = setTimeout(refreshList, refresh_timeout);
  }

  function refreshList(){
    $.ajax({
      url: list.data('url'),
      timeout: 5000,

      success: function(html){
        list.html(html);
        $('#queue-refresh-time').text(new Date().toLocaleTimeString());
      },

      complete: function(){
        scheduleListRefresh();
      }
    });
  }

  // --- Single Celery task state polling ---
  var info = $('#ctask-info');
  var state_timer = null;
  var state_done = false;

  function scheduleStateRefresh(){
    clearTimeout(state_timer);
    if (!document.hidden && !state_done)
      state_timer = setTimeout(refreshState, refresh_timeout);
  }

  function refreshState(){
    $.ajax({
      url: info.data('state-url'),
      dataType: 'json',
      timeout: 5000,

      success: function(json){
        $('#ctask-state').text(json.state);
        if (json.task_state)
          $('#task-state').text(json.task_state);
        if (json.chain_position)
          $('#chain-position').text(json.chain_position + '/' + json.chain_total);

        // Task finished, was revoked, or is no more linked to a running Django task
        if (json.ready || (json.task_id && !json.task_running)){
          state_done = true;
          $('#queue-actions').addClass('d-none');
          $('#ctask-finished').removeClass('d-none');
        }
      },

      complete: function(){
        scheduleStateRefresh();
      }
    });
  }

  if (list.length)
    scheduleListRefresh();
  if (info.length)
    scheduleStateRefresh();

  // Pause polling when the tab is not visible, refresh immediately when it is back
  document.addEventListener('visibilitychange', function(){
    if (document.hidden){
      clearTimeout(list_timer);
      clearTimeout(state_timer);
    } else {
      if (list.length)
        refreshList();
      if (info.length && !state_done)
        refreshState();
    }
  });

  // --- AJAX action buttons ---
  // Delegated so that it also works for buttons inside the refreshed list fragment
  $(document).on('click', '.queue-action-form button[type=submit]', function(ev){
    ev.preventDefault();

    var btn = $(this);
    var form = btn.closest('form');
    var confirmMsg = btn.data('confirm');

    if (confirmMsg && !window.confirm(confirmMsg))
      return;

    btn.prop('disabled', true);

    $.ajax({
      url: form.attr('action') || window.location.href,
      method: 'POST',
      dataType: 'json',
      data: {
        action: btn.attr('value'),
        csrfmiddlewaretoken: form.find('input[name=csrfmiddlewaretoken]').val()
      },

      success: function(json){
        showToast(json.message, 'success');
      },

      error: function(xhr){
        var msg = (xhr.responseJSON && xhr.responseJSON.message) || 'Action failed';
        showToast(msg, 'danger');
      },

      complete: function(){
        btn.prop('disabled', false);
        // Refresh the view right away to reflect the action
        if (list.length)
          refreshList();
        if (info.length && !state_done)
          refreshState();
      }
    });
  });
});
