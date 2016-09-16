x = rep(20,length.out=1000)+rnorm(1000,0,8)
y = rep(20,length.out=1000)+rnorm(1000,0,8)

png('regression_to_mean.png', w=520, h=520)
par(cex.lab =2.5, cex.axis=2, tck=.02)

plot(y~x)

idx = which(x>40)

for (k in idx)
    {   
    points(x[k],y[k], pch =21, bg="red")
    }

dev.off()

