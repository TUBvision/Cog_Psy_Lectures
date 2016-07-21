plot_duration <- function(parameter, my_ylim, label){

## lims
xvals = c(4,8,12)
ylims = my_ylim

part.col   = 'gray20'
surf.col   = 'gray40'
filled.col = 'gray60'
no_ind.col = 'gray90'


par(lwd=2, tck=.02, bty="n", cex=1.5, font.axis=3, font.lab=3, font=3, mar=c(3,3,1,0), mgp=c(1.2,0.3,0))
barplot(parameter, beside=TRUE, space = c(0.2, 1.2),  ylim=ylims, xpd=FALSE, ylab = label)


# surf.pch   = 22
# part.pch   = 21
# filled.pch = 24
# no_ind.pch = 25

# plot(xvals,   present['0',], type="p", pch=part.pch,   bg = 'lightgray', ylim=ylims, xlab="# items", ylab='duration', xaxp=c(4,12,2))
# points(xvals, present['1',], type="p", pch=surf.pch,   bg = "lightgray")
# points(xvals, present['2',], type="p", pch=filled.pch, bg = "lightgray")
# points(xvals, present['3',], type="p", pch=no_ind.pch, bg = "lightgray")
# 

legend(4, ylims[2], c("inducers", "subjective surface", "filled-in surface", "surface only"), pch = rep(22,4),  pt.bg=c(part.col, surf.col, filled.col, no_ind.col), bty="n")
}


## PRESENT ONLY
plotduration = function(mfix, label, my_ylim){
n=dim(mfix[,,1,1])[1]

# mean RT across subj: present; absent
surf=apply(mfix[,,2,1], 2, mean)
part=apply(mfix[,,2,2], 2, mean)

# calc sd: present; absent
sesurf=apply(mfix[,,2,1],2,sd)/sqrt(n)
separt=apply(mfix[,,2,2],2,sd)/sqrt(n)

# lims
ylims=my_ylim
xvals=c(4,8,12)

surf.pch=24
part.pch=21

par(lwd=2, tck=.02, bty="n", cex=1.5, font.axis=3, font.lab=3, font=3, mar=c(3,3,1,0), mgp=c(1.2,0.3,0))

plot(xvals,   surf, pch=surf.pch, type="p", ylim=ylims, xlab="# items", ylab=label, xaxp=c(4,12,2))
arrows(xvals, surf-sesurf, xvals, surf+sesurf, len=.2,ang=90, code=3)
arrows(xvals, part-separt, xvals, part+separt, len=.2,ang=90, code=3)

points(xvals, surf, type="p", pch=surf.pch, bg="white")
points(xvals, part, type="p", pch=part.pch, bg="white")

legend(4, ylims[2], c("surface","non-surface"), pch=c(surf.pch,part.pch), bty="n")
}

sess = 4
id = 'vf'
fix_type = 'dispersion'


## read and combine behavioral and eye movement data
eye_data = read.table(paste(id, '/', id, '_', sess, '_', fix_type, '.txt', sep=''), header=T)
log_data = read.table(paste(id, '/', id, '_', sess, sep=''), header=T)
log_data$correct = as.numeric(log_data$button-6 == log_data$search)


fix_per_trial = tapply(eye_data$fix_id+1, eye_data$trl, max)

data = c()
na_count = 0
eye_data$last = rep(0, length(eye_data$trl))


for (trl in row.names(fix_per_trial))
    {
    
    nfix_trl = fix_per_trial[trl]
    if (!is.na(nfix_trl))
        {
        curr_logdata  = log_data[as.numeric(trl),]
        curr_eyedata  = eye_data[eye_data$trl == as.numeric(trl),]
        
        curr_eyedata$last[nfix_trl] = 1
        
        row.names(curr_logdata) = c()
        ## repeat log_data row entry (= 1 trial) n_fix times
        if (nfix_trl != 1)
            {
            curr_logdata[2:nfix_trl,] = curr_logdata[1,]
            }
        curr_logdata  = cbind(curr_logdata, curr_eyedata[,2:7])
        data = rbind(data, curr_logdata)
        }
    else {na_count = na_count+1}
    }


## include only correct responses
data = subset(data, correct==1)
n_fix = tapply(data$fix_id, data.frame(data$target, data$nitems, data$search), max) + 1
X11()
plot_duration(n_fix[,,'0'], c(0, 30), 'number of fixations')
X11()
plot_duration(n_fix[,,'1'], c(0, 30), 'number of fixations')



## LAST fixation duration
sdata = subset(data, last == 1)
length(sdata$trl) == length(unique(sdata$trl))

last_fix = tapply(sdata$fdur, data.frame(sdata$target, sdata$nitems, sdata$search), mean)
X11()
plot_duration(last_fix[,,'0'], c(100, 400), 'last fixation duration')
X11()
plot_duration(last_fix[,,'1'], c(100, 400), 'last fixation duration')


## ALL but last fixation duration
sdata = subset(data, last != 1)

agg_sdata <-aggregate(sdata, by=list(trl), FUN=mean, na.rm=TRUE)

all_fix=tapply(agg_sdata$fdur, data.frame(agg_sdata$target, agg_sdata$nitems, agg_sdata$search), mean, na.rm=T)

X11()
plot_duration(all_fix[,,'0'], c(100, 400), 'fixation duration')
X11()
plot_duration(all_fix[,,'1'], c(100, 400), 'fixation duration')

