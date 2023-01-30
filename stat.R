library(ggplot2)

File = "20210412-promE-V105005 29C6d.mp4.csv"

frame = 30
Rsult = data.frame(Name=NA, Times=NA)
for(File in dir()){
	A <- read.csv(File, header = 0, sep=" ")

	Num = 0
	TMP = c()
	for (i in c(1:round(nrow(A)/frame,0))){
			Num = Num +1
			tmp = A[ (1 + (Num -1) * frame):(Num * frame), 2]
			TMP = c(TMP , sum(tmp))
	    }
	TMP_data <- data.frame(Name = File, Times = TMP)
	Rsult = rbind(Rsult, TMP_data)
}
Rsult = rbind(Rsult, TMP_data)
