'''
    The function for padding a batch of variable length sequences. Using this function we can construct small batches
    for training RNNs that require same length sequences. This version is not yet fully independent and while copying this
    function please take care of some of the basic dependencies should any errors arise.
'''

# write a function to pad the batch of sequences into a fixed length (by padding) numpy array
def pad(seqs):
    '''
        function to convert the list of seqs into a batch tensor (padding the batch to a nice length)
        @param
        seqs => the list of variable length scalar sequences
        @return => The batch tensor converted using the seqs
    '''

    lengths = map(lambda x: len(x), seqs) # extract the lengths of all the sequences in the batch
    max_length = max(lengths) # calculate the max of those lengths

    converted_seqs = [] # initialize it to empty list
    # for every sequence, pad it upto the length of max_length
    for seq in seqs:
        while(len(seq) != max_length):
            seq = seq + [PAD]
        # now append this list to the converted_seqs
        converted_seqs.append(seq)

    # return the numpy array corredponding to the converted_seqs
    return np.array(converted_seqs).T
    # The final transpose is because I prefer the Time_major format while training the RNNs
