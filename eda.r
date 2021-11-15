df = read.csv("data/train.csv")
df = as.data.frame(df)

summary(df)
str(df)
colnames(df)

as.factor(df$c1)

head(df)
df[,c(43,54)] = apply(df[,c(43:54)], 2, as.factor)

nrow(df)
ncol(df)

df.noNA = na.omit(df)
which(colnames(df) == c("c1"))
which(colnames(df) == c("c2"))


boxplot(df[, -c(1, 43, 44) ])
boxplot(df[43])
boxplot(df[44])
boxplot(df[45])
           