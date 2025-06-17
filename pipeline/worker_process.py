import numpy as np
import tensorflow as tf
import time
from queue import Queue
from threading import Thread
import pandas as pd

from .files import append_neurons_coords_to_csv
from .mask import filter_neurons_outside_mask



def filter_and_save_neurons(
        neurons,
        patch_metadata,
        dim_resolution,
        mask,
        include_neuron_mask_region_flag,
        vfv_shape,
        neurons_coords_csv_path
    ):
    """
    This function is responsible for taking the detected neurons on a patch
    and performing mask filtering over them, to check if they are outside the
    mask, and then saving the filters neurons coords at the given path csv file.
    If no neurons are present it saves a empty row with only the patch name as a 
    place holder for last processed patch.

    Parameters:
        neurons                         : A list containing all the neurons coords detected by the DoG model
        patch_metadata                  : A tuple containing patch metadata like the patch name and its offsets
        dim_resolution                  : The real scale of the patches
        mask                            : The acquistion's mask
        include_neuron_mask_region_flag : Boolean to specify if to include the mask regions of the detected neurons
        vfv_shape                       : The shape of the acquistion's Virtual Fused Volume
        neurons_coords_csv_path         : The path of the csv file that does/will contain the detected neurons coords
    """
    try:
        patch_name = patch_metadata['patch_name']
        patch_offset = patch_metadata['patch_offset']

        # set as a last processed patch placeholder
        detected_neurons_coords = pd.DataFrame(
            data=[[None, None, None, patch_name]],
            columns=["z", "y", "x", "patch_name"]
        )
        if include_neuron_mask_region_flag:
            detected_neurons_coords['mask_region'] = None

        # when there are neurons
        if neurons.size:
            # only take the neurson coordinates
            neurons_coords = neurons[:, :3]
            # add patch offsets to the neurons coords
            neurons_coords += patch_offset

            # if mask provided
            if mask is not None:
                # filter out the neurons outside the mask
                neurons_coords, neurons_mask_regions  = filter_neurons_outside_mask(
                    neurons_coords=neurons_coords,
                    mask=mask,
                    vfv_shape=vfv_shape
                )

            # after filtering, if there are neurons
            if neurons_coords.size:
                print(f"-+-+-+-+-+-+-+-+- Found {len(neurons_coords)} neurons in the patch {patch_name}")
                # scale the neurons coordinates back to their orignial scale
                neurons_coords *= dim_resolution
                
                # update the last processed patch placeholder
                # with the neurons coords found in that patch
                detected_neurons_coords = pd.DataFrame(
                    data=neurons_coords,
                    columns=["z", "y", "x"]
                )
                detected_neurons_coords['patch_name'] = patch_name
                if include_neuron_mask_region_flag:
                    # include the mask regions of the neurons
                    detected_neurons_coords['mask_region'] = neurons_mask_regions

        # add this last processed patch with the detected neurons coords (if any)
        # to the given path csv file. 
        append_neurons_coords_to_csv(
            neurons_coords=detected_neurons_coords,
            path=neurons_coords_csv_path
        )
    except Exception as exc:
        raise Exception(
            f"----------------- Error when filtering and saving the detected neurons.\nProcessed Patch Name: {patch_name}.\nException:\n {exc}"
        )



def gpu_task(
        patches_batch,
        unet,
        dog,
        dim_resolution,
        mask,
        include_neuron_mask_region_flag,
        vfv_shape,
        neurons_coords_csv_path
    ):
    """
    This function is responsible for taking a batch out from the queue and passing it
    to the gpu_task process.

    Parameters:
        patches_batch                   : The batch containing patches
        unet                            : The U-Net model
        dog                             : The DoG model
        dim_resolution                  : The real scale of the patches
        mask                            : The acquistion's mask
        include_neuron_mask_region_flag : Boolean to specify if to include the mask regions of the detected neurons
        vfv_shape                       : The shape of the acquistion's Virtual Fused Volume
        neurons_coords_csv_path         : The path of the csv file that does/will contain the detected neurons coords
    """
    try:
        # start time
        start = time.time()
        print(f"-------------- Starting GPU task")

        # getting the patches and their metadata from the batch
        patches, patches_metadata = zip(*patches_batch)

        # Initialising a valid batch input for the U-Net model.
        # The batch input is stack into 5D-tensor for the U-Net
        # (B, Z, Y, X, 1), where B is batch and 1 is a single channel.
        input = np.stack(patches, axis=0)[..., None]

        print(f"--------------- Performing U-Net Inference over a batch")
        # U-NET infernce
        preds = unet(input, training=False)
        preds = np.squeeze(preds, axis=-1)
        preds = tf.sigmoid(preds).numpy()*255

        # for every preditions made by the U-Net
        for pred_patch, patch_metadata in zip(preds, patches_metadata):
            print(f"---------------- Performing DoG on the patch {patch_metadata['patch_name']}")
            # performing DoG to get neurons
            neurons = dog.predict(pred_patch)

            filter_and_save_neurons(
                neurons=neurons, 
                patch_metadata=patch_metadata,
                dim_resolution=dim_resolution,
                mask=mask,
                include_neuron_mask_region_flag=include_neuron_mask_region_flag,
                vfv_shape=vfv_shape,
                neurons_coords_csv_path=neurons_coords_csv_path
            )

        end = time.time()
        print(f"------- GPU task total time taken: {end - start}")
    except Exception as exc:
        raise Exception(
            f"------- Error occured when performing the GPU task over a batch.\n{exc}"
        )


def gpu_worker(
        unet,
        dog,
        queue,
        dim_resolution,
        mask,
        include_neuron_mask_region_flag,
        vfv_shape,
        neurons_coords_csv_path
    ):
    """
    This function is responsible for taking a batch out from the queue and passing it
    to the gpu_task process.

    Parameters:
        unet                            : The U-Net model
        dog                             : The DoG model
        queue                           : The queue
        dim_resolution                  : The real scale of the patches
        mask                            : The acquistion's mask
        include_neuron_mask_region_flag : Boolean to specify if to include the mask regions of the detected neurons
        vfv_shape                       : The shape of the acquistion's Virtual Fused Volume
        neurons_coords_csv_path         : The path of the csv file that does/will contain the detected neurons coords
    """
    try:

        while True:
            # start time
            start = time.time()
            print(f"----------- GPU worker waiting!")

            # Get a batch out of the queue.
            patches_batch = queue.get()
            # end time
            end = time.time()
            # if the retrived batch is a null
            if patches_batch is None:
                # shut down gpu worker thread
                print(f"------------ GPU worker has finshed its work and is shutting down!")
                queue.task_done()
                return
            
            print(f"------------- Got batch in {end - start}, {queue.qsize()} batches left in the queue")

            # passing the batch to the gpu_task process, which performs 
            # U-Net, DoG and saves the detected neurons coords to the csv file.
            gpu_task(
                patches_batch=patches_batch,
                unet=unet,
                dog=dog,
                dim_resolution=dim_resolution,
                mask=mask,
                include_neuron_mask_region_flag=include_neuron_mask_region_flag,
                vfv_shape=vfv_shape,
                neurons_coords_csv_path=neurons_coords_csv_path
            )
            
            # marking that batch task to be done
            queue.task_done()
    except Exception as exc:
        raise Exception(
            f"----------- Error ocuured when the gpu worker was taking a batch out of the queue and passing it to the gpu task.\n{exc}"
        )



def queue_worker(patches, queue, batch_size):
    """
    This function is responsible for yielding a patch one by one of the patch generator and putting them inside a 
    the batch and then depending on the given size of the batch, when that condition satisfies it puts the batch in
    the queue.

    Parameters:
        patches    : The patch generator
        unet       : The queue
        batch_size : The size of the batch
    """
    try:
        # batch as a list to hold patches
        batch = list()
        # start time
        start = time.time()
        # for every yied patch of the generator
        for patch in patches: 
            # add the patch to the batch
            batch.append(patch)
            # if the batch size reach the given limit
            if (len(batch) == batch_size):
                # end time
                end = time.time()
                # add the batch to the queue as a tuple
                queue.put(tuple(batch))
                print(f"+++++++++++ Added a new batch of size {len(batch)} to the queue. Queue size now: {queue.qsize()}")
                print(f"+++++++++++ Total time it took to add the batch to the queue {end-start} sec.")
                # clear the current batch
                batch.clear()
                start = time.time()

        # if there is still something in the batch
        if batch:
            end = time.time()
            # add the batch to the queue
            queue.put(tuple(batch))
            print(f"++++++++++++ Added a new batch of size {len(batch)} to the queue. Queue size now: {queue.qsize()}")
            print(f"++++++++++++ Total time to add the batch to the queue {end-start} sec.")
            batch.clear()

        # put the end flag
        queue.put(None)
        print(f"+++++++++++++ Queue worker has finshed its work and is shutting down!")
    except Exception as exc:
        raise Exception(
            f"+++++++++++ Error occured when the queue worker was populating the queue.\n{exc}"
        )


def final_process(
        patches,
        unet,
        dog,
        batch_size,
        queue_size,
        dim_resolution,
        mask,
        include_neuron_mask_region_flag,
        vfv_shape,
        neurons_coords_csv_path
    ):
    """
    This function is responsible for conducting all the thread work, which involves initializing a queue, 
    initiating the queue_work thread, which adds batches to the queue, and the gpu_work thread, which is
    responsible for taking a batch from the queue and passing it to the gpu task (U-Net, Dog).

    Parameters:
        patches                         : The patch generator
        unet                            : The U-Net model
        dog                             : The DoG model
        batch_size                      : The size of the batch
        queue_size                      : The size of the queue
        dim_resolution                  : The real scale of the patches
        mask                            : The acquistion's mask
        include_neuron_mask_region_flag : Boolean to specify if to include the mask regions of the detected neurons
        vfv_shape                       : The shape of the acquistion's Virtual Fused Volume
        neurons_coords_csv_path         : The path of the csv file that does/will contain the detected neurons coords
    """
    try:
        # initialise the queue to hold batches
        queue = Queue(maxsize=queue_size)
        # initialise the gpu worker thread
        gpu_work = Thread(
            target=gpu_worker,
            args=(
                unet,
                dog,
                queue,
                dim_resolution,
                mask,
                include_neuron_mask_region_flag,
                vfv_shape,
                neurons_coords_csv_path
            )
        )
        # initialise the queue worker thread
        queue_work = Thread(
            target=queue_worker,
            args=(
                patches,
                queue,
                batch_size
            )
        )

        # start the threads
        queue_work.start()
        gpu_work.start()

        # wait for the threads and the queue to be done
        queue_work.join()
        queue.join()
        gpu_work.join()
    except Exception as exc:
        raise Exception(
            f"-++--++--++ Error occured during the final process.\n{exc}"
        )






