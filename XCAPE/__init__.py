"""
XCAPE Django Project Initializer
Includes runtime compatibility patches for Python 3.14+ and Django template contexts.
"""
import copy

try:
    from django.template import context as django_template_context

    def _safe_base_context_copy(self):
        duplicate = object.__new__(self.__class__)
        duplicate.__dict__.update(self.__dict__)
        duplicate.dicts = self.dicts[:]
        return duplicate

    def _safe_context_copy(self):
        duplicate = _safe_base_context_copy(self)
        duplicate.render_context = copy.copy(self.render_context)
        return duplicate

    # Apply patch to prevent Python 3.14 `super().__copy__()` AttributeError
    django_template_context.BaseContext.__copy__ = _safe_base_context_copy
    django_template_context.Context.__copy__ = _safe_context_copy
except Exception:
    pass
