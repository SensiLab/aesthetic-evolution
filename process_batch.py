"""
Batch processing for Qwen3-VL model using true batching on GPU.
@author: Stephen Krol 27/01/2026
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
from typing import List, Dict
import time

from utils import ComparisonJob


class Qwen3VLBatchProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-7B-Instruct", device: str = "cuda"):
        """
        Docstring for __init__
        @author: sjkrol
        
        :param model_name: Name of model to load from hugging face.
        :type model_name: str
        :param device: Device to load model onto (e.g., 'cuda' or 'cpu').
        :type device: str

        Return: None
        :rtype: None
        """
        self.device = device
        print(f"Loading model {model_name} on {device}...")
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        print("Model loaded successfully!")
        
    def process_batch_parallel(self, jobs: List[ComparisonJob], validate: bool = True) -> List[Dict]:
        """
        Docstring for process_batch_parallel. Method to process a batch of comparison jobs in parallel on GPU.
        @author: sjkrol

        :param jobs: List of ComparisonJob objects
        :type jobs: List[ComparisonJob]
        :param validate: Whether to perform validation checks
        :type validate: bool

        :return: List of results for each job. 
        :rtype: List[Dict]
        """
        print(f"Processing {len(jobs)} jobs in parallel on GPU...")
        start_time = time.time()
        
        try:
            # Prepare all messages for all jobs
            all_messages = []
            # all_images = []
            
            for job in jobs:
                # Load images for this job
                image1 = Image.open(job.image1_path)
                image2 = Image.open(job.image2_path)
                
                # Construct messages with only system prompt and images
                messages = [
                    {
                        "role": "system",
                        "content": job.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image1},
                            {"type": "image", "image": image2}
                        ]
                    }
                ]
                
                all_messages.append(messages)
                # all_images.extend([image1, image2])
            
            # Process all jobs together
            texts = []
            all_image_inputs = []
            all_video_inputs = []
            
            for messages in all_messages:
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)
                
            # Process vision info
            all_image_inputs, all_video_inputs = process_vision_info(all_messages)
            # all_image_inputs.extend(image_inputs)
            # all_video_inputs.extend(video_inputs if video_inputs else [])
            
            # Batch process all inputs together
            print("Preparing batch inputs...")
            inputs = self.processor(
                text=texts,
                images=all_image_inputs if all_image_inputs else None,
                videos=all_video_inputs if all_video_inputs else None,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            print(inputs["input_ids"].shape)

            # Validation checks
            if validate:
                print(f"\n{'='*60}")
                print("VALIDATION CHECKS:")
                print(f"{'='*60}")
                print(f"Number of jobs: {len(jobs)}")
                print(f"Number of text prompts: {len(texts)}")
                # print(f"Number of images loaded: {len(all_images)}")
                print(f"Expected images (2 per job): {len(jobs) * 2}")
                print(f"Batch size in inputs: {inputs.input_ids.shape[0]}")
                
                # Verify batch size matches number of jobs
                assert inputs.input_ids.shape[0] == len(jobs), \
                    f"Batch size mismatch! Expected {len(jobs)}, got {inputs.input_ids.shape[0]}"
                
                # Verify we have the right number of images
                # assert len(all_images) == len(jobs) * 2, \
                #     f"Image count mismatch! Expected {len(jobs) * 2}, got {len(all_images)}"
                
                print("✓ All validation checks passed!")
                print(f"{'='*60}\n")

            torch.cuda.synchronize()
            start = time.perf_counter()
            print(f"Generating responses for {len(jobs)} jobs in parallel...")
            # Generate all responses in parallel on GPU
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            end = time.perf_counter()
            print(f"Generation completed in {end - start:.2f}s")
            print(f"Time per job: {(end - start)/len(jobs):.2f}s")
            print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")    
            
            # Decode all outputs
            print("Decoding outputs...")
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            total_time = time.time() - start_time          
            # Prepare results
            results = []
            for i, (job, output_text) in enumerate(zip(jobs, output_texts)):
                results.append({
                    "job_id": job.job_id,
                    "status": "success",
                    "result": output_text,
                    "image1": job.image1_path,
                    "image2": job.image2_path
                })
                print(f"✓ Completed job {job.job_id}")
            
            print(f"\nTotal processing time: {total_time:.2f}s")
            print(f"Average time per job: {total_time/len(jobs):.2f}s")
            
            return results
            
        except Exception as e:
            print(f"✗ Batch processing failed: {e}")
            # Return error for all jobs
            return [
                {
                    "job_id": job.job_id,
                    "status": "error",
                    "error": str(e),
                    "image1": job.image1_path,
                    "image2": job.image2_path
                }
                for job in jobs
            ]
    
    def process_batch_chunked(self, jobs: List[ComparisonJob], chunk_size: int = 8) -> List[Dict]:
        """
        Docstring for process_batch_chunked. Process a large batch of comparison jobs in smaller chunks to avoid OOM.
        @author: sjkrol

        :param jobs: List of ComparisonJob Objects
        :type jobs: List[ComparisonJob]
        :param chunk_size: Max size of each batch chunk
        :type chunk_size: int

        :return: List of results for each job
        :rtype: List[Dict]
        """
        all_results = []
        total_jobs = len(jobs)
        
        print(f"Processing {total_jobs} jobs in chunks of {chunk_size}...")
        
        for i in range(0, total_jobs, chunk_size):
            chunk = jobs[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (total_jobs + chunk_size - 1) // chunk_size
            
            print(f"\n{'='*60}")
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} jobs)")
            print(f"{'='*60}")
            
            chunk_results = self.process_batch_parallel(chunk)
            all_results.extend(chunk_results)
            
            # Clear GPU cache between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_results


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = Qwen3VLBatchProcessor(
        model_name="Qwen/Qwen3-VL-7B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Define system prompt
    system_prompt = "You are an assisitant that compares two images and ranks which one is more aesthetically pleasing. You will be given two images and you need to output either '1' or '2' based on which image is more aesthetically pleasing."

    batch_size = 64
    jobs = []
    for i in range(batch_size):
        jobs.append(
            ComparisonJob(
                job_id=f"comparison_{i+1}",
                image1_path="curl_drawings_training_set_00001.png",
                image2_path="curl_drawings_training_set_00002.png",
                system_prompt=system_prompt
            )
        )
    
    # Process entire batch in parallel on GPU
    # For small batches with enough GPU memory:
    results = processor.process_batch_parallel(jobs)
    
    # For large batches, process in chunks to avoid OOM:
    # results = processor.process_batch_chunked(jobs, chunk_size=8)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    for result in results:
        print(f"\nJob ID: {result['job_id']}")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Result: {result['result']}")  # First 200 chars
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("-" * 80)