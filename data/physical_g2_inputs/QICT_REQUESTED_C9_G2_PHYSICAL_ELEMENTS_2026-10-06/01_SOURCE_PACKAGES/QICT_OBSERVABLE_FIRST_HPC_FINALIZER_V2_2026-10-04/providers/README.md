# Physical provider drop-in directory

Place one certified provider at each required path:

- `L24/GLOBAL_CHARGED_UCONE_PROVIDER.py`
- `L32/GLOBAL_CHARGED_UCONE_PROVIDER.py`
- `L48/GLOBAL_CHARGED_UCONE_PROVIDER.py`
- `L64/GLOBAL_CHARGED_UCONE_PROVIDER.py`
- `L96/GLOBAL_CHARGED_UCONE_PROVIDER.py`
- `L128/GLOBAL_CHARGED_UCONE_PROVIDER.py`

Start from `../templates/GLOBAL_CHARGED_UCONE_PROVIDER_TEMPLATE.py` but do not set `physical_payload=true` until the metadata claims are backed by the required Kato/Riesz/C10/rank-six certificates.
