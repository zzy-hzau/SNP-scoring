"""Assembly identity is part of every coordinate-based input."""
import re

def canonical_assembly(value):
 text=str(value or '').upper()
 for pattern,name in [(r'SL\s*4[._]0','SL4.0'),(r'SL\s*5[._]0','SL5.0'),(r'IRGSP','IRGSP-1.0'),(r'AGPV3','AGPv3.25'),(r'CRI|TM.?1','TM-1_CRI'),(r'FRONTIER','CDC_Frontier')]:
  if re.search(pattern,text):return name
 return text

def require_same_assembly(source,target):
 a,b=canonical_assembly(source),canonical_assembly(target)
 if not a or not b:raise ValueError('Reference assembly must be specified for coordinate-based tools')
 if a!=b:raise ValueError(f'Coordinate mapping required: {source} → {target}; no verified mapping supplied')
