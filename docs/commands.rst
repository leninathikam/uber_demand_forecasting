Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Syncing data locally
^^^^^^^^^^^^^^^^^^^^

* `make sync_data_to_local` will copy the contents of `data/` into `local-data-cache/`.
* `make sync_data_from_local` will restore the contents of `local-data-cache/` back into `data/`.
