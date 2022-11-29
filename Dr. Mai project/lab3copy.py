sc = spark.sparkContext

url1 = sc.textFile("dbfs:///FileStore/tables/shortLab3data1-3.txt")
url2 = sc.textFile("dbfs:///FileStore/tables/shortLab3data0-2.txt")

print(url1.collect())




def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

"""dbutils.fs.mkdirs("FileStore/tables/lab3short")
dbutils.fs.cp("dbfs:///FileStore/tables/shortLab3data0.txt", "FileStore/tables/lab3short")
dbutils.fs.cp("dbfs:///FileStore/tables/shortLab3data1.txt", "FileStore/tables/lab3short")
urlRDD=sc.textFile("dbfs:///FileStore/tables/shortLab3data0-3.txt")
print(urlRDD.collect())"""

[('www.example1.com', ['www.example14.com', 'www.example14.com', 'www.example16.com', 'www.example3.com', 'www.example4.com', 'www.example7.com', 'www.example9.com']), ('www.example10.com', ['www.example20.com']), ('www.example11.com', ['www.example8.com']), ('www.example12.com', ['www.example19.com', 'www.example4.com']), ('www.example13.com', ['www.example5.com']), ('www.example14.com', ['www.example10.com', 'www.example11.com', 'www.example3.com', 'www.example7.com']), ('www.example15.com', ['www.example1.com', 'www.example1.com', 'www.example18.com']), ('www.example16.com', ['www.example12.com']), ('www.example18.com', ['www.example15.com', 'www.example2.com', 'www.example4.com', 'www.example7.com']), ('www.example19.com', ['www.example2.com', 'www.example5.com']), ('www.example2.com', ['www.example3.com']), ('www.example20.com', ['www.example14.com', 'www.example17.com', 'www.example5.com']), ('www.example4.com', ['www.example16.com', 'www.example17.com', 'www.example17.com', 'www.example6.com']), ('www.example5.com', ['www.example1.com', 'www.example20.com', 'www.example3.com']), ('www.example6.com', ['www.example12.com', 'www.example13.com']), ('www.example7.com', ['www.example12.com', 'www.example16.com']), ('www.example8.com', ['www.example12.com', 'www.example4.com'])]

sc = spark.sparkContext

url1A = sc.textFile("dbfs///FileStore/tables/fullLab3data0-3.txt")
url2A = sc.textFile("dbfs:///FileStore/tables/shortLab3data1.txt")
url3A = sc.textFile("dbfs:///FileStore/tables/shortLab3data2.txt")
url4A = sc.textFile("dbfs:///FileStore/tables/shortLab3data3.txt")

url5A=url1A+url2A+url3A+url4A

def splitter(str):
    return str.split()

pairs2 = url5A.map(lambda x: (x.split(" ",1)[0], x.split(" ",1)[1]))

splitPairs2= pairs2.flatMapValues(splitter).sortByKey()

rddSwitched2 = splitPairs2.map(lambda x: (x[1], x[0]))

groupedRdd2=rddSwitched2.groupByKey().map(lambda x : (x[0], list(x[1]))).sortByKey()
print(groupedRdd2.take(10))
print(groupedRdd2.count())
