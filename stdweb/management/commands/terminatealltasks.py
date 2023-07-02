from django.core.management.base import BaseCommand

from stdweb import celery

class Command(BaseCommand):
    help = 'Terminate all queued tasks'

    def handle(self, *args, **options):
        inspect = celery.app.control.inspect()

        for queues in (inspect.active(), inspect.reserved(), inspect.scheduled()):
            for task_list in queues.values():
                for t in task_list:
                    if t['id']:
                        print("Revoking and terminating task %s at %s" % (t['id'], t['hostname']))
                        celery.app.control.revoke(t['id'], terminate=True, signal='SIGKILL')
