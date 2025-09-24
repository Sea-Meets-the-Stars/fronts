
from fronts.finding import dev as finding_dev


if __name__ == "__main__":

    # Vanilla
    #finding_dev.run_a_test('vanilla')#, tst_idx=(0,500,700))

    # Add thin 
    #finding_dev.run_a_test('thin')#, tst_idx=(0,500,700))

    # Remove weak
    #finding_dev.run_a_test('rm_weak')#, tst_idx=(0,500,700))

    # Dilate
    finding_dev.run_a_test('rm_weak-thin-dilate')#, tst_idx=(0,500,700))