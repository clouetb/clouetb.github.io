# -*- coding:utf-8 -*-
from mako import runtime, filters, cache
UNDEFINED = runtime.UNDEFINED
STOP_RENDERING = runtime.STOP_RENDERING
__M_dict_builtin = dict
__M_locals_builtin = locals
_magic_number = 10
_modified_time = 1741477172.401427
_enable_loop = True
_template_filename = '/Users/benoitclouet/.local/pipx/venvs/nikola/lib/python3.13/site-packages/nikola/data/themes/base/templates/pagination_helper.tmpl'
_template_uri = 'pagination_helper.tmpl'
_source_encoding = 'utf-8'
_exports = ['page_navigation']


def render_body(context,**pageargs):
    __M_caller = context.caller_stack._push_frame()
    try:
        __M_locals = __M_dict_builtin(pageargs=pageargs)
        __M_writer = context.writer()
        __M_writer('\n')
        return ''
    finally:
        context.caller_stack._pop_frame()


def render_page_navigation(context,current_page,page_links,prevlink,nextlink,prev_next_links_reversed,surrounding=5):
    __M_caller = context.caller_stack._push_frame()
    try:
        len = context.get('len', UNDEFINED)
        enumerate = context.get('enumerate', UNDEFINED)
        abs = context.get('abs', UNDEFINED)
        __M_writer = context.writer()
        __M_writer('\n<div class="page-navigation">\n')
        for i, link in enumerate(page_links):
            pass
            if abs(i - current_page) <= surrounding or i == 0 or i == len(page_links) - 1:
                pass
                if i == current_page:
                    __M_writer('        <span class="current-page">')
                    __M_writer(str(i+1))
                    __M_writer('</span>\n')
                else:
                    __M_writer('        <a href="')
                    __M_writer(str(page_links[i]))
                    __M_writer('">')
                    __M_writer(str(i+1))
                    __M_writer('</a>\n')
            elif i == current_page - surrounding - 1 or i == current_page + surrounding + 1:
                __M_writer('      <span class="ellipsis">…</span>\n')
        __M_writer('</div>\n')
        return ''
    finally:
        context.caller_stack._pop_frame()


"""
__M_BEGIN_METADATA
{"filename": "/Users/benoitclouet/.local/pipx/venvs/nikola/lib/python3.13/site-packages/nikola/data/themes/base/templates/pagination_helper.tmpl", "uri": "pagination_helper.tmpl", "source_encoding": "utf-8", "line_map": {"16": 0, "21": 16, "27": 2, "34": 2, "35": 4, "37": 5, "39": 6, "40": 7, "41": 7, "42": 7, "43": 8, "44": 9, "45": 9, "46": 9, "47": 9, "48": 9, "49": 11, "50": 12, "51": 15, "57": 51}}
__M_END_METADATA
"""
