from datasets import load_dataset
import random
import argparse
import csv


def main(args):
    unshuffled_dataset= load_dataset("NbAiLab/NCC", streaming=True, split='train',use_auth_token=True)
    dataset = unshuffled_dataset.shuffle(buffer_size=10_000, seed=42)
    
    sample_id = 0
    sample_length = 512
    max_samples_from_doc = 3
    total_samples = 1_200_000
    trans_chars = "'\",.:;-_*?/\n"

    with open(args.output_file, 'wt') as outfile:
        tsv_writer = csv.writer(outfile, delimiter='\t')

        for doc in dataset:
            text = doc['text']
            lang = doc['lang_fasttext']
            docid = doc['id']
            publish_year = int(doc['publish_year'])

            if lang == "no" and publish_year>=1990:
                words = text.split()
                split_doc_words = []
                split_doc_words = [words[x:x+sample_length] for x in range(0, len(words), sample_length)]
                
                # Choose method
                r = random.randint(1,100)
                uncased = True if r <= 70 else False
                r = random.randint(1,100)
                nopunc = True if r <= 50 else False
                r = random.randint(1,100)
                nospace = True if r <= 30 else False
                
                for ex in split_doc_words[:max_samples_from_doc]:
                    target = (' '.join(ex))
                    target = target.replace("\t"," ")
                    
                    source = target
                    method = "t"

                    if uncased == True:
                        source = source.lower()
                        method += "u" 

                    if nopunc == True:
                        trans_table = source.maketrans("", "", trans_chars)
                        source = source.translate(trans_table)
                        method += "p"

                    if nospace == True:
                        source = source.replace(" ","")
                        method += "s"

                    sample_id+=1
                    tsv_writer.writerow([str(sample_id), method, source, target])
                    #print(f"{str(sample_id)}\t{method}\t{source}\t{target}")
            
            if sample_id >= total_samples:
                print(f"Finished printing: {sample_id} samples!")
                break


def parse_args():
    # Parse commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file', required=True, help='Output file. Will overwrite it exists')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)


