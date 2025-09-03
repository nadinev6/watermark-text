#!/usr/bin/env python3
"""
Test script for watermarking API endpoints.

This script tests the watermarking functionality by making HTTP requests
to the local API server and validating the responses.
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running and healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ API Health Check: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"  Status: {health_data.get('status', 'unknown')}")
            return True
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ API Health Check Failed: {e}")
        return False

def test_watermarking_methods():
    """Test getting watermarking methods information."""
    try:
        response = requests.get(f"{BASE_URL}/api/watermark/methods", timeout=10)
        print(f"✓ Watermarking Methods: {response.status_code}")
        
        if response.status_code == 200:
            methods_data = response.json()
            print(f"  Supported methods: {methods_data.get('supported_methods', [])}")
            print(f"  Visibility options: {methods_data.get('visibility_options', [])}")
            return methods_data
        else:
            print(f"  Error: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"✗ Watermarking Methods Failed: {e}")
        return None

def test_watermark_embedding():
    """Test watermark embedding functionality."""
    try:
        # Test data
        test_text = """
        This is a sample document that will be used to test the watermarking functionality.
        The text needs to be long enough to support reliable watermarking operations.
        We're testing the stegano LSB method which hides information in the least significant bits.
        This should preserve the readability while embedding the watermark invisibly.
        """
        
        watermark_content = "© 2024 Test Author - Document ID: TEST123"
        
        # Embed watermark request
        embed_request = {
            "text": test_text.strip(),
            "watermark_content": watermark_content,
            "method": "stegano_lsb",
            "visibility": "hidden",
            "preserve_formatting": True
        }
        
        print("Testing watermark embedding...")
        response = requests.post(
            f"{BASE_URL}/api/watermark/embed",
            json=embed_request,
            timeout=30
        )
        
        print(f"✓ Watermark Embedding: {response.status_code}")
        
        if response.status_code == 200:
            embed_data = response.json()
            watermarked_text = embed_data.get("watermarked_text", "")
            watermark_hash = embed_data.get("watermark_hash", "")
            processing_time = embed_data.get("processing_time_ms", 0)
            
            print(f"  Watermark hash: {watermark_hash[:16]}...")
            print(f"  Processing time: {processing_time}ms")
            print(f"  Original length: {len(test_text)}")
            print(f"  Watermarked length: {len(watermarked_text)}")
            print(f"  Text appears identical: {test_text.strip() == watermarked_text}")
            
            # Check if text looks the same (it should for stegano)
            if test_text.strip() != watermarked_text:
                print(f"  ⚠️  Text content changed (expected for stegano LSB)")
            
            return {
                "watermarked_text": watermarked_text,
                "watermark_hash": watermark_hash,
                "original_watermark": watermark_content
            }
        else:
            print(f"  Error: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Watermark Embedding Failed: {e}")
        return None

def test_watermark_extraction(watermarked_text: str, expected_watermark: str):
    """Test watermark extraction functionality."""
    try:
        # Extract watermark request
        extract_request = {
            "text": watermarked_text,
            "methods": ["stegano_lsb", "visible_text"]
        }
        
        print("Testing watermark extraction...")
        response = requests.post(
            f"{BASE_URL}/api/watermark/extract",
            json=extract_request,
            timeout=30
        )
        
        print(f"✓ Watermark Extraction: {response.status_code}")
        
        if response.status_code == 200:
            extract_data = response.json()
            watermark_found = extract_data.get("watermark_found", False)
            extracted_content = extract_data.get("watermark_content", "")
            extraction_method = extract_data.get("extraction_method", "")
            confidence_score = extract_data.get("confidence_score", 0.0)
            processing_time = extract_data.get("processing_time_ms", 0)
            
            print(f"  Watermark found: {watermark_found}")
            print(f"  Extraction method: {extraction_method}")
            print(f"  Confidence score: {confidence_score}")
            print(f"  Processing time: {processing_time}ms")
            
            if watermark_found:
                print(f"  Extracted content: '{extracted_content}'")
                print(f"  Expected content: '{expected_watermark}'")
                print(f"  Content matches: {extracted_content == expected_watermark}")
                
                return {
                    "success": True,
                    "extracted_content": extracted_content,
                    "matches_expected": extracted_content == expected_watermark
                }
            else:
                print(f"  ⚠️  No watermark found in text")
                return {"success": False, "reason": "no_watermark_found"}
        else:
            print(f"  Error: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Watermark Extraction Failed: {e}")
        return None

def test_watermark_validation(watermarked_text: str, expected_watermark: str):
    """Test watermark validation functionality."""
    try:
        # Validation request
        validation_request = {
            "watermarked_text": watermarked_text,
            "expected_watermark": expected_watermark
        }
        
        print("Testing watermark validation...")
        response = requests.post(
            f"{BASE_URL}/api/watermark/validate",
            json=validation_request,
            timeout=30
        )
        
        print(f"✓ Watermark Validation: {response.status_code}")
        
        if response.status_code == 200:
            validation_data = response.json()
            is_valid = validation_data.get("is_valid", False)
            integrity_score = validation_data.get("integrity_score", 0.0)
            extracted_watermark = validation_data.get("extracted_watermark", "")
            processing_time = validation_data.get("processing_time_ms", 0)
            
            print(f"  Validation result: {is_valid}")
            print(f"  Integrity score: {integrity_score:.3f}")
            print(f"  Processing time: {processing_time}ms")
            
            if extracted_watermark:
                print(f"  Extracted during validation: '{extracted_watermark}'")
            
            return {
                "is_valid": is_valid,
                "integrity_score": integrity_score
            }
        else:
            print(f"  Error: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Watermark Validation Failed: {e}")
        return None

def test_watermarking_stats():
    """Test watermarking service statistics."""
    try:
        response = requests.get(f"{BASE_URL}/api/watermark/stats", timeout=10)
        print(f"✓ Watermarking Stats: {response.status_code}")
        
        if response.status_code == 200:
            stats_data = response.json()
            embedding_stats = stats_data.get("embedding_stats", {})
            extraction_stats = stats_data.get("extraction_stats", {})
            
            print(f"  Total embeds: {embedding_stats.get('total_embeds', 0)}")
            print(f"  Successful embeds: {embedding_stats.get('successful_embeds', 0)}")
            print(f"  Total extractions: {extraction_stats.get('total_extractions', 0)}")
            print(f"  Successful extractions: {extraction_stats.get('successful_extractions', 0)}")
            
            return stats_data
        else:
            print(f"  Error: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Watermarking Stats Failed: {e}")
        return None

def test_visible_watermarking():
    """Test visible watermarking method."""
    try:
        test_text = "This is a test document for visible watermarking."
        watermark_content = "© 2024 Test Author"
        
        # Test visible watermark embedding
        embed_request = {
            "text": test_text,
            "watermark_content": watermark_content,
            "method": "visible_text",
            "visibility": "visible",
            "preserve_formatting": True
        }
        
        print("Testing visible watermark embedding...")
        response = requests.post(
            f"{BASE_URL}/api/watermark/embed",
            json=embed_request,
            timeout=30
        )
        
        if response.status_code == 200:
            embed_data = response.json()
            watermarked_text = embed_data.get("watermarked_text", "")
            
            print(f"✓ Visible Watermark Embedded")
            print(f"  Original: '{test_text}'")
            print(f"  Watermarked: '{watermarked_text}'")
            
            # Test extraction
            extract_request = {
                "text": watermarked_text,
                "methods": ["visible_text"]
            }
            
            extract_response = requests.post(
                f"{BASE_URL}/api/watermark/extract",
                json=extract_request,
                timeout=30
            )
            
            if extract_response.status_code == 200:
                extract_data = extract_response.json()
                extracted_content = extract_data.get("watermark_content", "")
                
                print(f"✓ Visible Watermark Extracted: '{extracted_content}'")
                print(f"  Matches original: {extracted_content == watermark_content}")
                
                return True
        
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Visible Watermarking Test Failed: {e}")
        return False

def main():
    """Run comprehensive watermarking API tests."""
    print("=" * 60)
    print("AI Watermark Detection Tool - Watermarking API Tests")
    print("=" * 60)
    
    # Test 1: API Health
    if not test_api_health():
        print("❌ API is not running. Please start the backend server first.")
        print("   Run: cd backend && python main.py")
        return
    
    print()
    
    # Test 2: Get methods information
    methods_info = test_watermarking_methods()
    if not methods_info:
        print("❌ Could not retrieve watermarking methods")
        return
    
    print()
    
    # Test 3: Stegano watermarking (embed + extract + validate)
    print("Testing Stegano LSB Watermarking...")
    embed_result = test_watermark_embedding()
    
    if embed_result:
        print()
        
        # Test extraction
        extract_result = test_watermark_extraction(
            embed_result["watermarked_text"],
            embed_result["original_watermark"]
        )
        
        if extract_result and extract_result.get("success"):
            print()
            
            # Test validation
            validation_result = test_watermark_validation(
                embed_result["watermarked_text"],
                embed_result["original_watermark"]
            )
            
            if validation_result:
                print(f"✓ End-to-end stegano watermarking test completed successfully!")
            else:
                print("❌ Watermark validation failed")
        else:
            print("❌ Watermark extraction failed")
    else:
        print("❌ Watermark embedding failed")
    
    print()
    
    # Test 4: Visible watermarking
    print("Testing Visible Text Watermarking...")
    visible_result = test_visible_watermarking()
    
    if visible_result:
        print("✓ Visible watermarking test completed successfully!")
    else:
        print("❌ Visible watermarking test failed")
    
    print()
    
    # Test 5: Service statistics
    stats_result = test_watermarking_stats()
    
    print()
    print("=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    if embed_result and extract_result and validation_result and visible_result and stats_result:
        print("🎉 All watermarking API tests passed!")
        print()
        print("Next steps:")
        print("1. Integrate watermarking controls into the frontend TextEditor")
        print("2. Add watermark display to the ResultsPanel")
        print("3. Test the complete user workflow")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("   Make sure the backend server is running with: python backend/main.py")

if __name__ == "__main__":
    main()