# Ground truth — Porto da Cruz baptisms, 1863

`portodacruz_batismos_1863.csv` is the reference set used to evaluate the nine runs reported in the RecordIndex 2.0 article (three models × three pipeline modes).

## Source

Parish register of baptisms of Porto da Cruz (Madeira, Portugal), year 1863, held by the Arquivo Regional e Biblioteca Pública da Madeira, reference code `PT/ABM/PMCH04/001/00022`. The page images are freely accessible at <https://arquivo-abm.madeira.gov.pt/descriptions/4741>.

The CSV was built from the archive's own descriptive index of the register (one entry per baptism), with no manual annotation by the authors.

## Format

132 lines, UTF-8, one baptism entry per line, no header:

```
<archival ID>; Registo de batismo n.º <k>: <name>. Pai: <father>; Mãe: <mother>; <date YYYY-MM-DD>
```

Example:

```
PT/ABM/PMCH04/001/00022/000001; Registo de batismo n.º 1: Maria. Pai: João Francisco de Gouveia; Mãe: Eulália Vieira; 1863-01-01
```

The fields used by the evaluation are name, father, mother and date. The text is in Portuguese, as in the archival index.

## Use

Send it as `reference_csv` to the evaluation service (`POST /evaluate`, port 8001), together with the pipeline output JSON and `collection_type=batismo`. See the "Módulo de avaliação" section of the main README.

SHA-256: `1f7d8627366b28dfba882a0abcff436dabbe7ea02537a48d84d214e4b9a1216b`
