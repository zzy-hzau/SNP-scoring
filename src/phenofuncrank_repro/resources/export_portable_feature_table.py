#!/usr/bin/env python3
"Export portable feature table."
import argparse
import json
from pathlib import Path,PurePosixPath,PureWindowsPath
import pyarrow as pa
import pyarrow.parquet as pq


def relative_resource(value,markers):
    if value is None:return None
    text=str(value).replace("\\","/")
    if not (PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()):
        return text
    parts=text.split("/")
    matches=[i for i,p in enumerate(parts) if p in markers]
    if not matches:
        raise ValueError("Absolute provenance path lacks a declared resource marker")
    relative="/".join(parts[matches[-1]+1:])
    if not relative or ".." in relative.split("/"):
        raise ValueError("Provenance reference is not a safe relative resource")
    return relative


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--report",required=True,type=Path)
    parser.add_argument("--path-columns",nargs="+",default=["prot_safe_msa_file"])
    parser.add_argument("--resource-markers",nargs="+",
                        default=["msa_untrimmed_from_orthogroup_sequences","msa"])
    args=parser.parse_args()
    source=pq.ParquetFile(args.input)
    if args.output.exists():raise FileExistsError("Use a new exported resource path")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    writer=None;rows=0;scientific_columns=[]
    try:
        for batch in source.iter_batches(batch_size=100000):
            table=pa.Table.from_batches([batch]).replace_schema_metadata(None)
            original=table
            for name in args.path_columns:
                if name not in table.column_names:continue
                idx=table.column_names.index(name)
                field=table.schema.field(idx)
                values=[relative_resource(value,args.resource_markers)
                        for value in table[name].to_pylist()]
                table=table.set_column(idx,field,pa.array(values,type=field.type))
            scientific_columns=[c for c in table.column_names if c not in args.path_columns]
            if not original.select(scientific_columns).equals(
                    table.select(scientific_columns),check_metadata=False):
                raise ValueError("Scientific columns changed during provenance export")
            if writer is None:
                writer=pq.ParquetWriter(args.output,table.schema,compression="zstd")
            writer.write_table(table);rows+=table.num_rows
    finally:
        if writer is not None:writer.close()
    exported=pq.ParquetFile(args.output)
    compared=0
    import itertools
    for left,right in itertools.zip_longest(
        source.iter_batches(batch_size=100000,columns=scientific_columns),
        exported.iter_batches(batch_size=100000,columns=scientific_columns)):
        if left is None or right is None:raise ValueError("Output batch length mismatch")
        if not pa.Table.from_batches([left]).equals(pa.Table.from_batches([right]),
                                                    check_metadata=False):
            raise ValueError("Persisted scientific values/identity/type changed")
        compared+=left.num_rows
    paths_relative=True
    for batch in exported.iter_batches(batch_size=100000,
                columns=[c for c in args.path_columns if c in exported.schema.names]):
        for col in batch.columns:
            for value in col.to_pylist():
                if value is None:continue
                if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
                    paths_relative=False
    report={"status":"PASSED" if paths_relative and compared==rows else "FAILED",
            "scope":"Actual complete processed feature table privacy/relocation export: "
                    "declared provenance paths relative, all scientific arrays/order/"
                    "types/missing values independently reread and exactly preserved.",
            "rows":rows,"scientific_columns":scientific_columns,
            "path_columns":args.path_columns,"all_exported_paths_relative":paths_relative,
            "schema_metadata_removed":True,"scientific_values_exact":True,
            "source_or_output_machine_paths_recorded_in_public_report":False}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    if report["status"]!="PASSED":raise SystemExit(1)


if __name__=="__main__":
    main()
