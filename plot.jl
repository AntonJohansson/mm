using Plots, DelimitedFiles
plotly()

data, header = readdlm(ARGS[1], header=true)

p = plot(size=(1920,1080))
for i=2:size(data,2)
    plot!(data[:,1], data[:,i], label=header[i])
end
display(p)
