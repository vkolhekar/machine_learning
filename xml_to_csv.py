import xml.etree.cElementTree as et
import pandas as pd
 
def getvalueofnode(node):
    """ return node text or None """
    return node.text if node is not None else None
 
 
def main():
    """ main """
    parsed_xml = et.parse("UNdata_Export_20200309_072349660.xml")
    dfcols = ['Country or Area', 'Year', 'Sex', 'Age','Cause of death (WHO)','Record Type','Reliability','Source Year','Value','Value Footnotes']
    df_xml = pd.DataFrame(columns=dfcols)
 
    for node in parsed_xml.getroot():
        record = node.attrib.get('None')
        #print(record)
        country = node.find('Country or Area')
        year = node.find('Year')
        sex = node.find('Sex')
        age = node.find('Age')
        cause = node.find('Cause of death (WHO)')
        rec_type = node.find('Record Type')
        rel = node.find('Reliability')
        source = node.find('Source Year')
        val = node.find('Value')
        val_footnotes = node.find('Value Footnotes')

        df_xml = df_xml.append(
            pd.Series([record,
                       getvalueofnode(country), 
                       getvalueofnode(year), 
                       getvalueofnode(sex),
                       getvalueofnode(age),
                       getvalueofnode(cause),
                       getvalueofnode(rec_type),
                       getvalueofnode(rel),
                       getvalueofnode(source),
                       getvalueofnode(val),
                       getvalueofnode(val_footnotes)], index=dfcols),
            ignore_index=True)

    print (df_xml)
 
main()