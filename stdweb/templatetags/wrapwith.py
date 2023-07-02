# Copyright (c) 2020, DabApps
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Original at https://github.com/dabapps/django-wrapwith/tree/master

from django.conf import settings
from django.template import Library
from django.template.loader_tags import do_include

register = Library()

"""
The {% wrapwith %} template tag is exactly like {% include %}, but it's a block tag. The
rendered contents of the block up to {% endwrapwith %} are injected into the included
template as a variable called {{ wrapped }}.

This code is structured to reuse the built-in {% include %} tag implementation as much
as possible, to avoid copy-pasting duplicated code.
"""


class RenderNodelistVariable:
    """
    Quacks like a template.Variable, but wraps a nodelist which is rendered when the
    variable is resolved. Used to inject the rendered nodelist into the
    included wrapper template.
    """

    def __init__(self, nodelist):
        self.nodelist = nodelist

    def resolve(self, context):
        return self.nodelist.render(context)


class ResolveWithAliases:
    """
    Wraps a FilterExpression and injects the WRAPWITH_TEMPLATES alias
    dictionary into its context before resolving the variable name.
    """

    def __init__(self, template):
        self.template = template
        self.aliases = getattr(settings, "WRAPWITH_TEMPLATES", {})

    def resolve(self, context):
        with context.push(self.aliases):
            return self.template.resolve(context)


@register.tag(name="wrapwith")
def do_wrapwith(parser, token):
    """
    Calls the do_include function, but injects the contents of the block into
    the extra_context of the included template.
    """
    include_node = do_include(parser, token)
    nodelist = parser.parse(("endwrapwith",))
    parser.delete_first_token()
    include_node.template = ResolveWithAliases(include_node.template)
    include_node.extra_context["wrapped"] = RenderNodelistVariable(nodelist)
    return include_node
