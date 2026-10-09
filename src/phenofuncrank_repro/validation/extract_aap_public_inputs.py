"""Read the published AAP sample/PC block without changing the workbook."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

N = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def read_cells(path):
    with zipfile.ZipFile(path) as z:
        shared = [''.join(t.text or '' for t in s.findall('.//m:t',N)) for s in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        workbook = ET.fromstring(z.read('xl/workbook.xml'))
        sheet = next(s for s in workbook.find('m:sheets',N) if s.attrib['name']=='Figure 5')
        rels = {r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        target = rels[sheet.attrib[f'{{{R}}}id']]
        target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
        result = {}
        for c in ET.fromstring(z.read(target)).findall('.//m:sheetData/m:row/m:c',N):
            v = c.find('m:v',N);kind = c.attrib.get('t')
            if kind=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//m:t',N))
            elif v is None:continue
            elif kind=='s':value=shared[int(v.text)]
            elif kind in ['str','e']:value=v.text
            else:value=float(v.text)
            result[c.attrib['r']]=value
        return result


def extract(source_data, output_directory):
    fields = ['sample_id','genotype','hundred_seed_weight_g','PC1','PC2','PC3','PC4','PC5']
    cells = read_cells(source_data)
    candidates = [int(key[1:]) for key,value in cells.items() if key.startswith('A') and value=='sample_id']
    row = next(r for r in candidates if [cells.get(f'{col}{r}') for col in 'ABCDEFGH']==fields)
    first = row+1;samples=[];pcs=[]
    while cells.get(f'A{row+1}') is not None:
        row+=1
        values=[cells.get(f'{col}{row}') for col in 'ABCDEFGH']
        samples.append(values[:3]);pcs.append([values[0]]+values[3:])
    output_directory=Path(output_directory);output_directory.mkdir(parents=True,exist_ok=True)
    for filename,header,rows in [('samples.tsv',fields[:3],samples),('PCs.tsv',['sample_id']+fields[3:],pcs)]:
        with (output_directory/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(header);writer.writerows(rows)
    receipt={'source_workbook_sha256':hashlib.sha256(Path(source_data).read_bytes()).hexdigest(),'sheet':'Figure 5','data_range':f'A{first}:H{row}','subjects':len(samples),'source_modified':False}
    (output_directory/'extraction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();extract(args.source_data,args.output_directory)


if __name__=='__main__':
    main()
