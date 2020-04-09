import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from Bio import SeqIO


class ErrorProb:

    def __init__(self, reads, skip_bad=False, bad_pc=0.5):
        self.reads = reads
        self.skip_bad = skip_bad
        self.bad_pc = bad_pc
        self._max_len = None
        self._error_prob = None
        self._quality = None
        self.raw_qualities = None

    def count_n(self, x):
        return x.count('N') / len(x)

    @property
    def max_len(self):
        if self._max_len is None:
            self._max_len = 0
            for fastq in self.reads:
                if self.skip_bad and self.count_n(fastq.seq) >= self.bad_pc:
                    continue
                if len(fastq.seq) > self._max_len:
                    self._max_len = len(fastq.seq)
        return self._max_len

    def q(self, qualities):
        return np.pad(np.power(10, (- np.array(qualities) / 10)), (0, self.max_len - len(qualities))), \
               np.pad(np.array([1 for _ in qualities]), (0, self.max_len - len(qualities)))

    @property
    def error_prob(self):
        if self._error_prob is None:
            qualities, total, self.raw_qualities = [], [], []
            for fastq in self.reads:
                if self.skip_bad and self.count_n(fastq.seq) >= self.bad_pc:
                    continue
                current_quality = fastq.letter_annotations['phred_quality']
                padded = self.q(current_quality)
                qualities.append(padded[0])
                total.append(padded[1])
                self.raw_qualities.append(np.pad(current_quality, (0, self.max_len - len(current_quality))))
            self._quality = np.sum(qualities, axis=0)
            self._error_prob = self._quality / np.sum(total, axis=0)
            self.raw_qualities = np.sum(self.raw_qualities, axis=0) / np.sum(total, axis=0)
        return self._error_prob

    def plot_error_prob(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
        ax.plot([i for i in range(1, self.max_len + 1)], self.error_prob)
        ax.set_title('Mean error per position')
        ax.set_xlabel('Position')
        ax.set_ylabel('Mean error')

    def plot_quality(self, ax=None):
        _ = self.error_prob
        if ax is None:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
        ax.plot([i for i in range(1, self.max_len + 1)], self.raw_qualities)
        ax.set_title('Mean quality per position')
        ax.set_xlabel('Position')
        ax.set_ylabel('Mean quality')


class GCContent:

    def __init__(self, reads, no_n=False, skip_bad=False, bad_pc=0.5):
        self.reads = reads
        self._gc_content = None
        self.no_n = no_n
        self.skip_bad = skip_bad
        self.bad_pc = bad_pc

    def count_n(self, x):
        return x.count('N') / len(x)

    @property
    def gc_content(self):
        if self._gc_content is None:
            self._gc_content = []
            for fastq in self.reads:
                if self.skip_bad and self.count_n(fastq.seq) >= self.bad_pc:
                    continue
                self._gc_content.append(
                    sum([self.gc(x) for x in fastq.seq]) / self.read_len(fastq.seq, no_n=self.no_n))
        return self._gc_content

    def read_len(self, seq, no_n=False):
        r_len = len(seq)
        if no_n:
            r_len -= seq.count('N')
        return r_len

    def gc(self, x):
        if x in ['C', 'G']:
            return 1
        return 0

    def plot_gc(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
        content = Counter(sorted(self.gc_content))
        ax.plot(*map(list, [content.keys(), content.values()]), 'bo',
                 *map(list, [content.keys(), content.values()]))

        ax.set_title('GC content')
        ax.set_xlabel('GC content, share')
        ax.set_ylabel('Read count')


class FastQRead(GCContent, ErrorProb):

    def __init__(self, filename, no_n=False, skip_bad=False, bad_pc=0.5):
        self.filename = filename
        self._gc_content = None
        self.no_n = no_n
        self.skip_bad = skip_bad
        self.bad_pc = bad_pc
        self.reads = self.get_reads()
        GCContent.__init__(self, self.reads, no_n=no_n, skip_bad=skip_bad, bad_pc=bad_pc)
        ErrorProb.__init__(self, self.reads, skip_bad=skip_bad, bad_pc=bad_pc)

    def get_reads(self):
        try:
            return self.read_file()
        except ValueError:
            self.read_bad_file()
            return self.read_file()

    def read_file(self):
        reads = []
        for record in SeqIO.parse(self.filename, 'fastq'):
            reads.append(record)
        return reads

    def read_bad_file(self):
        data = []
        i = 1
        with open(self.filename) as f:
            current = []
            for line in f.read().splitlines():
                current.append(line)
                if not i % 4:
                    data.append(current)
                    current = []
                i += 1

        new_data = []

        for read in data:
            seq, quality = read[1], read[3]
            if len(seq) != len(quality):
                continue
            first_title = read[0][1:]
            second_title = read[2][1:]
            if first_title != second_title:
                if second_title.startswith(first_title):
                    second_title = first_title
                else:
                    continue
            new_data.extend(['@' + first_title, seq, '+' + second_title, quality])

        with open(self.filename, 'w') as f:
            for item in new_data:
                f.write(f'{item}\n')


