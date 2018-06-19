#=
Potential world representations:
1- 3x20 where each column represents a block as xyz
2- 1x20x3 special representation for the conv architecture.
    xyz coordinates are input channels. However, the spatial
    dimension does not reflect the reel world relations (left of, right of)
3- Grid representation
    x, y, z, 20 where each block is located in the grid depending on its coordinates
    and the input channels represent the block ids
4- Raw images
=#

mutable struct BlockInstance
    world
    source
    sourceind
    target
    sentence
    img
    sentence_str
    imgmat
end

mutable struct Minibatch
    worlds
    sources
    sourceinds
    targets
    sentences
    samplesize
    imgs
    sentence_strs
    imgmats
end

function read_instances(fname; imgdict=nothing)
    instances = BlockInstance[]
    data = readdlm(fname)
    data2 = readdlm(string(fname, ".human"), quotes=false)
    path = split(fname, "/")
    path[end-1] = contains(path[end-1], "Blank") ? "SRD_Blank" : "SRD"
    data3 = readdlm(join(path, "/"))
    maxind = 0
    for i=1:size(data, 1)
        world = convert(Array{Float32}, reshape(data[i, 7:66], 3, 20))
        source = convert(Array{Float32}, data[i, 1:3])
        sourceind = data3[i, 1] + 1
        target = convert(Array{Float32}, data[i, 4:6])
        sentence = Int[]
        for j=67:size(data, 2)
            if data[i,j] == ""
                break
            end
            push!(sentence, data[i,j])
            maxind = data[i,j] > maxind ? data[i,j] : maxind
        end
        m = imgdict != nothing ? imgdict[data2[i, 7]] : nothing
        push!(instances, BlockInstance(world, source, sourceind, target, reverse(sentence), data2[i, 7], strip(join(data2[i, 8:end], " ")), m))
    end

    return instances, maxind
end

function minibatch(instances, vocabsize, bs; sortinstances=true, useimgs=false, addworld=false)
    sorted = sortinstances ? sort(instances; by=x->length(x.sentence)) : instances
    batches = Minibatch[]
    worlddim = addworld ? 9 : 3
    for i=1:bs:length(sorted)
        samples = sorted[i:min(i+bs-1, length(sorted))]
        longest = length(samples[end].sentence)
        ins = ones(Int, length(samples), longest) * vocabsize
        world = zeros(Float32, length(samples)*20, worlddim)
        source = zeros(Float32, length(samples), 3)
        sourceind = Integer[]
        target = zeros(Float32, length(samples), 3)
        mats = useimgs ? zeros(Float32, 120, 120, 1, length(samples)) : nothing
        imgs = Any[]
        strs = Any[]

        for j=1:length(samples)
            ins[j, (longest-length(samples[j].sentence)+1):end] = samples[j].sentence
            
            #blocks*batchsize x 3
            #first blocks, second blocks, ...
            for k=1:20
                if addworld
                    x,y,z = samples[j].world[:, k]
                    e1 = abs(-1 - x)
                    e2 = abs(1 - x)
                    e3 = abs(0 - y)
                    e4 = abs(1 - y)
                    e5 = abs(-1 - z)
                    e6 = abs(1 - z)
                    world[(k-1)*length(samples)+j, :] = reshape(Float32[x,y,z,e1,e2,e3,e4,e5,e6], 1, worlddim)
                else
                    world[(k-1)*length(samples)+j, :] = samples[j].world[:, k]'
                end
            end
            
            source[j, :] = samples[j].source
            push!(sourceind, samples[j].sourceind)
            target[j, :] = samples[j].target
            if useimgs
                mats[:, :, 1, j] = samples[j].imgmat
            end
            push!(imgs, samples[j].img)
            push!(strs, samples[j].sentence_str)
        end
        m = Minibatch(world, source, sourceind, target, ins, length(samples), imgs, strs, mats)

        push!(batches, m)
    end
    return batches
end

function minibatch_vae(instances, vocabsize, bs; sortinstances=true, useimgs=false, addworld=false)
    sorted = sortinstances ? sort(instances; by=x->length(x.sentence)) : instances
    batches = Minibatch[]
    worlddim = 3
    for i=1:bs:length(sorted)
        samples = sorted[i:min(i+bs-1, length(sorted))]
        longest = length(samples[end].sentence)
        ins = ones(Int, length(samples), longest) * vocabsize
        world = zeros(Float32, 20 * worlddim, length(samples))
        source = zeros(Float32, length(samples), 3)
        sourceind = Integer[]
        target = zeros(Float32, length(samples), 3)
        mats = useimgs ? zeros(Float32, 120, 120, 1, length(samples)) : nothing
        imgs = Any[]
        strs = Any[]

        for j=1:length(samples)
            ins[j, (longest-length(samples[j].sentence)+1):end] = samples[j].sentence
            world[:, j] = reshape(samples[j].world, 60, 1)
            
            source[j, :] = samples[j].source
            push!(sourceind, samples[j].sourceind)
            target[j, :] = samples[j].target
            if useimgs
                mats[:, :, 1, j] = samples[j].imgmat
            end
            push!(imgs, samples[j].img)
            push!(strs, samples[j].sentence_str)
        end
        m = Minibatch(world, source, sourceind, target, ins, length(samples), imgs, strs, mats)

        push!(batches, m)
    end
    return batches
end

function read_dataset(foldername, bs=32; imgparent=nothing, addworld=false,
                      minibatchf=minibatch)
    trnd, devd, tstd = (nothing, nothing, nothing)
    if imgparent != nothing
        trnd = load(string(imgparent, "trainset_imgs.jld"), "imgs")
        devd = load(string(imgparent, "devset_imgs.jld"), "imgs")
        tstd = load(string(imgparent, "testset_imgs.jld"), "imgs")
    end

    trn, maxind_trn = read_instances(string(foldername, "Train.mat"); imgdict=trnd)
    dev, maxind_dev = read_instances(string(foldername, "Dev.mat"); imgdict=devd)
    tst, maxind_tst = read_instances(string(foldername, "Test.mat"); imgdict=tstd)

    vocabsize = maximum([maxind_trn, maxind_dev, maxind_tst]) + 1

    trnb, devb, tstb = map(x->minibatchf(x, vocabsize, bs; useimgs=(imgparent != nothing), addworld=addworld), [trn, dev, tst])
    return (trnb, devb, tstb, vocabsize)
end
