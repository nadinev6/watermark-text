import React from 'react';
import { 
  FileText, 
  Type, 
  AlignLeft, 
  Search,
  Settings,
  ToggleLeft,
  ToggleRight,
  Eye,
  EyeOff,
  Bold,
  Italic,
  Underline,
  AlignCenter,
  AlignRight,
  AlignJustify,
  List,
  ListOrdered,
  FolderOpen,
  Shield,
  ShieldCheck,
  Download,
  Upload
} from 'lucide-react';
import { ExpandableTabs } from './ExpandableTabs';
import { DropdownMenu } from './DropdownMenu';
import './TextEditor.css';
import './ExpandableTabs.css';
import './DropdownMenu.css';

interface TextEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  onAnalyze?: () => void;
  isAnalyzing?: boolean;
  watermarkInfo?: {
    hasWatermark: boolean;
    extractedContent?: string;
    confidence?: number;
    method?: string;
  };
  onWatermarkChange?: (info: {
    hasWatermark: boolean;
    extractedContent?: string;
    confidence?: number;
    method?: string;
  }) => void;
}

interface AnalysisSettings {
  autoDetection: boolean;
  showConfidenceScoring: boolean;
  showDetailedAnalysis: boolean;
}

interface WatermarkSettings {
  method: 'stegano_lsb' | 'visible_text';
  visibility: 'hidden' | 'visible';
  content: string;
  preserveFormatting: boolean;
}

interface FormattingState {
  bold: boolean;
  italic: boolean;
  underline: boolean;
  alignment: 'left' | 'center' | 'right' | 'justify';
  listType: 'none' | 'bullet' | 'numbered';
}

export const TextEditor: React.FC<TextEditorProps> = ({ 
  value, 
  onChange, 
  placeholder = "Start typing...",
  onAnalyze,
  isAnalyzing = false,
  watermarkInfo,
  onWatermarkChange
}) => {
  const editorRef = React.useRef<HTMLDivElement>(null);
  const [wordCount, setWordCount] = React.useState(0);
  const [charCount, setCharCount] = React.useState(0);
  const [isInitialized, setIsInitialized] = React.useState(false);

  // Watermarking state
  const [watermarkSettings, setWatermarkSettings] = React.useState<WatermarkSettings>({
    method: 'stegano_lsb',
    visibility: 'hidden',
    content: '',
    preserveFormatting: true
  });
  const [isWatermarking, setIsWatermarking] = React.useState(false);

  // Sample watermarked texts for testing
  const sampleWatermarkedTexts = [
    {
      title: "AI Ethics Discussion",
      content: `The rapid advancement of artificial intelligence has brought both unprecedented opportunities and significant challenges to our society. As we stand at the crossroads of technological innovation, it becomes increasingly important to establish ethical frameworks that guide the development and deployment of AI systems.

One of the most pressing concerns is the potential for AI to perpetuate or amplify existing biases present in training data. When machine learning models are trained on historical data that reflects societal inequalities, they may inadvertently learn to discriminate against certain groups. This phenomenon has been observed in various applications, from hiring algorithms that favor certain demographics to facial recognition systems that perform poorly on individuals with darker skin tones.

Furthermore, the question of transparency and explainability in AI decision-making processes remains a critical issue. Many modern AI systems, particularly deep learning models, operate as "black boxes" where the reasoning behind their decisions is not easily interpretable by humans. This lack of transparency can be problematic in high-stakes scenarios such as medical diagnosis, criminal justice, or financial lending, where understanding the rationale behind AI recommendations is crucial for accountability and trust.

The development of AI governance frameworks must also address the concentration of power in the hands of a few large technology companies. As AI capabilities become more sophisticated and expensive to develop, there is a risk that the benefits of AI will be unevenly distributed, potentially exacerbating existing inequalities rather than helping to address them.`
    },
    {
      title: "Climate Change Solutions",
      content: `Climate change represents one of the most significant challenges facing humanity in the 21st century. The scientific consensus is clear: human activities, particularly the emission of greenhouse gases from fossil fuel combustion, are driving unprecedented changes in Earth's climate system. However, alongside this sobering reality comes a growing recognition that innovative solutions and collective action can still make a meaningful difference.

Renewable energy technologies have experienced remarkable progress in recent years, with solar and wind power becoming increasingly cost-competitive with traditional fossil fuels. The dramatic reduction in the cost of solar panels, coupled with improvements in energy storage systems, has made clean energy more accessible to communities worldwide. Countries like Denmark and Costa Rica have demonstrated that it is possible to generate significant portions of their electricity from renewable sources while maintaining economic growth.

Beyond energy production, the concept of circular economy principles offers promising pathways for reducing waste and resource consumption. By designing products for durability, repairability, and recyclability, we can minimize the environmental impact of manufacturing and consumption. Companies across various industries are beginning to adopt these principles, recognizing that sustainable practices can also drive innovation and create new business opportunities.

Individual actions, while important, must be complemented by systemic changes in policy, infrastructure, and economic incentives. Carbon pricing mechanisms, investment in public transportation, and support for sustainable agriculture are examples of policy interventions that can accelerate the transition to a low-carbon economy.`
    },
    {
      title: "Future of Remote Work",
      content: `The global shift toward remote work, accelerated by the COVID-19 pandemic, has fundamentally transformed how we think about employment, productivity, and work-life balance. What began as an emergency response to public health concerns has evolved into a permanent feature of the modern workplace, with far-reaching implications for individuals, organizations, and society as a whole.

One of the most significant benefits of remote work is the increased flexibility it offers to employees. Workers can better balance their professional responsibilities with personal commitments, potentially leading to improved mental health and job satisfaction. Parents, in particular, have found that remote work allows them to be more present for their families while still maintaining their careers. Additionally, the elimination of daily commutes has resulted in time savings and reduced transportation costs for many workers.

From an organizational perspective, remote work has opened up access to a global talent pool. Companies are no longer limited to hiring within their immediate geographic area and can recruit the best candidates regardless of location. This has been particularly beneficial for specialized roles where local talent may be scarce. Furthermore, many organizations have reported maintained or even increased productivity levels among remote workers, challenging traditional assumptions about the need for physical presence in the office.

However, the transition to remote work has also presented challenges that require careful consideration and management. Social isolation and the blurring of boundaries between work and personal life have emerged as significant concerns for many remote workers. The lack of spontaneous interactions and informal communication that naturally occur in physical offices can impact team cohesion and innovation.`
    }
  ];

  const openSampleDocument = (sample: { title: string; content: string }) => {
    onChange(sample.content);
    setActiveDropdown(null);
    
    // Optional: Show a brief notification about the loaded document
    // You could add a toast notification here if you have a notification system
  };
  const [activeDropdown, setActiveDropdown] = React.useState<string | null>(null);
  const [selectedTabIndex, setSelectedTabIndex] = React.useState<number | null>(null);
  const [analysisSettings, setAnalysisSettings] = React.useState<AnalysisSettings>({
    autoDetection: false,
    showConfidenceScoring: true,
    showDetailedAnalysis: true,
  });
  const [formatting, setFormatting] = React.useState<FormattingState>({
    bold: false,
    italic: false,
    underline: false,
    alignment: 'left',
    listType: 'none',
  });


  // Initialize editor content
  React.useEffect(() => {
    if (editorRef.current && !isInitialized) {
      editorRef.current.innerHTML = value || '';
      setIsInitialized(true);
    }
  }, [value, isInitialized]);

  // Update editor content when value prop changes
  React.useEffect(() => {
    if (editorRef.current && isInitialized) {
      const currentContent = editorRef.current.innerHTML;
      if (currentContent !== value) {
        editorRef.current.innerHTML = value || '';
      }
    }
  }, [value, isInitialized]);

  // Update word and character count
  React.useEffect(() => {
    const textContent = editorRef.current?.textContent || '';
    const words = textContent.trim() ? textContent.trim().split(/\s+/).length : 0;
    setWordCount(words);
    setCharCount(textContent.length);
  }, [value]);

  const handleInput = () => {
    if (editorRef.current) {
      const content = editorRef.current.innerHTML;
      onChange(content);
      
      // Update word and character count
      const textContent = editorRef.current.textContent || '';
      const words = textContent.trim() ? textContent.trim().split(/\s+/).length : 0;
      setWordCount(words);
      setCharCount(textContent.length);
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLDivElement>) => {
    e.preventDefault();
    
    // Get plain text from clipboard
    const pastedText = e.clipboardData.getData('text/plain');
    
    // Insert as plain text to maintain consistency
    document.execCommand('insertText', false, pastedText);
  };

  const handleAnalyze = () => {
    const textContent = editorRef.current?.textContent || '';
    if (onAnalyze && textContent.trim()) {
      onAnalyze();
      setActiveDropdown(null);
    }
  };

  const toggleAnalysisSetting = (setting: keyof AnalysisSettings) => {
    setAnalysisSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }));
  };

  const applyFormatting = (type: string) => {
    if (!editorRef.current) return;
    
    editorRef.current.focus();
    
    switch (type) {
      case 'bold':
        document.execCommand('bold', false);
        setFormatting(prev => ({ ...prev, bold: !prev.bold }));
        break;
        
      case 'italic':
        document.execCommand('italic', false);
        setFormatting(prev => ({ ...prev, italic: !prev.italic }));
        break;
        
      case 'underline':
        document.execCommand('underline', false);
        setFormatting(prev => ({ ...prev, underline: !prev.underline }));
        break;
        
      case 'bullet-list':
        document.execCommand('insertUnorderedList', false);
        setFormatting(prev => ({ 
          ...prev, 
          listType: prev.listType === 'bullet' ? 'none' : 'bullet' 
        }));
        break;
        
      case 'numbered-list':
        document.execCommand('insertOrderedList', false);
        setFormatting(prev => ({ 
          ...prev, 
          listType: prev.listType === 'numbered' ? 'none' : 'numbered' 
        }));
        break;
    }
    
    // Update content after formatting
    setTimeout(() => {
      if (editorRef.current) {
        onChange(editorRef.current.innerHTML);
      }
    }, 0);
  };

  const setAlignment = (alignment: FormattingState['alignment']) => {
    if (!editorRef.current) return;
    
    editorRef.current.focus();
    
    switch (alignment) {
      case 'left':
        document.execCommand('justifyLeft', false);
        break;
      case 'center':
        document.execCommand('justifyCenter', false);
        break;
      case 'right':
        document.execCommand('justifyRight', false);
        break;
      case 'justify':
        document.execCommand('justifyFull', false);
        break;
    }
    
    setFormatting(prev => ({ ...prev, alignment }));
    setActiveDropdown(null);
    
    // Update content after alignment change
    setTimeout(() => {
      if (editorRef.current) {
        onChange(editorRef.current.innerHTML);
      }
    }, 0);
  };

  const embedWatermark = async () => {
    if (!value.trim() || !watermarkSettings.content.trim()) {
      alert('Please provide both text and watermark content');
      return;
    }

    setIsWatermarking(true);
    try {
      const response = await fetch('/api/watermark/embed', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: editorRef.current?.textContent || value,
          watermark_content: watermarkSettings.content,
          method: watermarkSettings.method,
          visibility: watermarkSettings.visibility,
          preserve_formatting: watermarkSettings.preserveFormatting
        }),
      });

      if (response.ok) {
        const result = await response.json();
        onChange(result.watermarked_text);
        const newWatermarkInfo = {
          hasWatermark: true,
          extractedContent: watermarkSettings.content,
          confidence: 1.0,
          method: watermarkSettings.method
        };
        onWatermarkChange?.(newWatermarkInfo);
        alert('Watermark embedded successfully!');
      } else {
        const error = await response.json();
        alert(`Watermark embedding failed: ${error.detail?.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Watermark embedding error:', error);
      alert('Failed to embed watermark. Please try again.');
    } finally {
      setIsWatermarking(false);
      setActiveDropdown(null);
    }
  };

  const extractWatermark = async () => {
    if (!value.trim()) {
      alert('Please provide text to extract watermark from');
      return;
    }

    setIsWatermarking(true);
    try {
      const response = await fetch('/api/watermark/extract', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: editorRef.current?.textContent || value,
          methods: ['stegano_lsb', 'visible_text']
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setWatermarkStatus({
          hasWatermark: result.watermark_found,
          extractedContent: result.watermark_content,
          confidence: result.confidence_score
        });
        
        if (result.watermark_found) {
          alert(`Watermark found: "${result.watermark_content}"`);
        } else {
          alert('No watermark detected in this text');
        }
      } else {
        const error = await response.json();
        alert(`Watermark extraction failed: ${error.detail?.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Watermark extraction error:', error);
      alert('Failed to extract watermark. Please try again.');
    } finally {
      setIsWatermarking(false);
      setActiveDropdown(null);
    }
  };

  const exportWatermarkedText = () => {
    const textContent = editorRef.current?.textContent || value;
    if (!textContent.trim()) {
      alert('No text to export');
      return;
    }

    const blob = new Blob([textContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `watermarked-document-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setActiveDropdown(null);
  };

  const importText = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.txt,.md';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string;
          onChange(content);
        };
        reader.readAsText(file);
      }
    };
    input.click();
    setActiveDropdown(null);
  };

  const handleTabClick = (tabName: string, tabIndex: number) => {
    console.log('Tab clicked:', tabName, tabIndex); // Debug log
    console.log('Current activeDropdown:', activeDropdown);
    if (activeDropdown === tabName) {
      console.log('Closing dropdown');
      setActiveDropdown(null);
      setSelectedTabIndex(null);
    } else {
      console.log('Opening dropdown:', tabName);
      setActiveDropdown(tabName);
      setSelectedTabIndex(tabIndex);
    }
  };

  const tabs = [
    { 
      title: "Document", 
      icon: FileText
    },
    { 
      title: "Format", 
      icon: Type
    },
    { 
      title: "Layout", 
      icon: AlignLeft
    },
    { 
      title: "Watermark", 
      icon: Shield
    },
    { type: "separator" as const },
    { 
      title: "Actions", 
      icon: Settings
    },
  ];

  const getDropdownItems = (type: string) => {
    switch (type) {
      case 'document':
        console.log('Building document menu, sample texts:', sampleWatermarkedTexts);
        return [
          { 
            label: 'New Document', 
            icon: FileText, 
            onClick: () => {
              onChange('');
              setActiveDropdown(null);
            }
          },
          { type: 'separator' as const },
          { 
            label: 'Open Sample Documents', 
            icon: FolderOpen,
            type: 'submenu' as const,
            submenu: [
              {
                label: 'Human-Written Article',
                icon: FileText,
                onClick: () => {
                  onChange(`The Art of Coffee: A Personal Journey

I've been drinking coffee for over twenty years, and I can honestly say that my relationship with this magical bean has evolved dramatically. What started as a necessity during late-night college study sessions has transformed into a genuine passion for understanding the nuances of flavor, origin, and brewing methods.

My morning routine begins at 6:30 AM with the gentle whir of my burr grinder. I prefer single-origin beans from small roasters who work directly with farmers. There's something deeply satisfying about knowing the story behind your cup – the altitude where the beans grew, the processing method used, even the name of the farmer who cultivated them.

The ritual itself has become meditative. Water heated to exactly 205°F, a 1:16 ratio of coffee to water, and a slow, deliberate pour that takes about four minutes from start to finish. My friends think I'm obsessive, but they don't understand that this isn't just about caffeine – it's about starting the day with intention and mindfulness.

Last month, I visited a coffee farm in Guatemala. Walking through the rows of coffee plants, seeing the careful hand-picking process, and meeting the families who dedicate their lives to this craft gave me a profound appreciation for every cup I drink. The farmer, Carlos, showed me how to identify perfectly ripe cherries and explained how weather patterns affect the final flavor profile.

Coffee has taught me patience, attention to detail, and the value of craftsmanship. It's connected me with people from different cultures and backgrounds. Most importantly, it's reminded me that the best things in life require time, care, and respect for the process.`);
                }
              },
              {
                label: 'AI-Generated Content',
                icon: FileText,
                onClick: () => {
                  onChange(`The Future of Artificial Intelligence in Healthcare

Artificial intelligence is revolutionizing the healthcare industry at an unprecedented pace. Machine learning algorithms are now capable of analyzing medical images with accuracy that rivals experienced radiologists, while natural language processing systems can extract valuable insights from vast amounts of clinical documentation.

One of the most promising applications of AI in healthcare is predictive analytics. By analyzing patient data patterns, AI systems can identify individuals at risk of developing chronic conditions before symptoms appear. This proactive approach enables healthcare providers to implement preventive measures, ultimately reducing costs and improving patient outcomes.

Diagnostic accuracy has improved significantly with AI integration. Deep learning models trained on millions of medical images can detect subtle abnormalities that might be missed by human observers. For instance, AI systems have demonstrated remarkable success in identifying early-stage cancers, diabetic retinopathy, and cardiovascular diseases.

The implementation of AI-powered chatbots and virtual assistants has streamlined patient interactions and reduced administrative burdens. These systems can handle routine inquiries, schedule appointments, and provide basic medical information, allowing healthcare professionals to focus on more complex patient care tasks.

However, the integration of AI in healthcare also presents challenges. Data privacy concerns, algorithmic bias, and the need for regulatory compliance must be carefully addressed. Healthcare organizations must ensure that AI systems are transparent, explainable, and aligned with ethical standards.

As we move forward, the collaboration between AI technology and human expertise will be crucial for maximizing benefits while maintaining the personal touch that is essential in healthcare delivery.`);
                }
              },
              {
                label: 'Academic Paper Excerpt',
                icon: FileText,
                onClick: () => {
                  onChange(`Abstract

This study examines the impact of remote work policies on employee productivity and job satisfaction in the post-pandemic era. Through a comprehensive analysis of 1,247 employees across 15 industries, we investigated the correlation between work-from-home arrangements and key performance indicators.

Introduction

The COVID-19 pandemic fundamentally altered the landscape of modern work, forcing organizations worldwide to rapidly adopt remote work policies. What began as an emergency response has evolved into a permanent shift in how we conceptualize the workplace. This research addresses the critical question: How do remote work arrangements affect long-term employee productivity and satisfaction?

Methodology

We conducted a mixed-methods study combining quantitative surveys with qualitative interviews. Participants were recruited from companies that implemented remote work policies between March 2020 and December 2022. Data collection occurred over an 18-month period, allowing us to capture both immediate and sustained effects of remote work implementation.

Our survey instrument included validated scales for measuring job satisfaction (Job Descriptive Index), productivity metrics (self-reported and supervisor-evaluated), and work-life balance indicators. Semi-structured interviews were conducted with a subset of 127 participants to provide deeper insights into the quantitative findings.

Results

The data revealed a complex relationship between remote work and employee outcomes. While 73% of participants reported increased job satisfaction, productivity measures showed more nuanced results. Task-oriented roles demonstrated a 12% increase in productivity, while collaborative positions showed a 7% decrease in certain metrics.

Interestingly, employees with dedicated home office spaces reported significantly higher satisfaction scores (M = 4.2, SD = 0.8) compared to those working from shared spaces (M = 3.1, SD = 1.2), t(1245) = 8.7, p < 0.001.`);
                }
              },
              {
                label: 'Creative Writing Sample',
                icon: FileText,
                onClick: () => {
                  onChange(`The Last Bookstore

Maya pushed open the heavy wooden door, and the familiar scent of aged paper and leather bindings enveloped her like an old friend's embrace. Dust motes danced in the afternoon sunlight that streamed through the tall windows of Henderson's Books, casting golden rectangles across the worn hardwood floors.

"We're closing next month," Mr. Henderson said without looking up from the ledger spread across his cluttered desk. His voice carried the weight of forty-three years spent among these shelves, forty-three years of recommending the perfect book to the perfect person at the perfect moment.

Maya's heart sank. She'd been coming here since she was seven, when her grandmother first brought her to this magical place tucked between a dry cleaner and a small café on Maple Street. This was where she'd discovered Matilda, where she'd fallen in love with Jane Austen, where she'd spent countless Saturday afternoons lost in worlds that existed only between covers.

"But why?" she asked, though she already knew the answer. The same reason all the small bookstores were disappearing – online retailers, e-books, changing habits, rising rents.

Mr. Henderson finally looked up, his eyes tired behind wire-rimmed glasses. "People don't browse anymore, Maya. They know exactly what they want before they walk in, if they walk in at all. There's no serendipity left in book buying."

Maya wandered through the narrow aisles, her fingers trailing along familiar spines. In the poetry section, she found a slim volume of Pablo Neruda that she'd been meaning to buy for months. In fiction, a first edition of "To Kill a Mockingbird" sat waiting for someone who would appreciate its significance.

"What will happen to all the books?" she whispered.

"Some will go to other stores, some to collectors. The rest..." He shrugged, a gesture that spoke of resignation and heartbreak in equal measure.

That night, Maya couldn't sleep. She kept thinking about all the stories that would be scattered to the wind, all the perfect matches between reader and book that would never happen. By morning, she had made a decision that would change everything.`);
                }
              },
              {
                label: 'Technical Documentation',
                icon: FileText,
                onClick: () => {
                  onChange(`API Authentication Guide

Overview
This document provides comprehensive instructions for implementing authentication in our REST API. All API endpoints require proper authentication to ensure data security and access control.

Authentication Methods

1. API Key Authentication
The simplest method for server-to-server communication. Include your API key in the request header:

Header: Authorization: Bearer YOUR_API_KEY

Example:
curl -H "Authorization: Bearer sk_live_abc123xyz789" \\
     https://api.example.com/v1/users

2. OAuth 2.0 Flow
For applications requiring user consent and token-based authentication:

Step 1: Redirect users to authorization endpoint
GET https://api.example.com/oauth/authorize?
    client_id=YOUR_CLIENT_ID&
    response_type=code&
    redirect_uri=YOUR_CALLBACK_URL&
    scope=read write

Step 2: Exchange authorization code for access token
POST https://api.example.com/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=AUTHORIZATION_CODE&
client_id=YOUR_CLIENT_ID&
client_secret=YOUR_CLIENT_SECRET&
redirect_uri=YOUR_CALLBACK_URL

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "def50200..."
}

Error Handling

401 Unauthorized: Invalid or missing authentication credentials
403 Forbidden: Valid credentials but insufficient permissions
429 Too Many Requests: Rate limit exceeded

Rate Limiting
- Free tier: 100 requests per hour
- Pro tier: 1,000 requests per hour
- Enterprise: Custom limits available

Best Practices
1. Store API keys securely using environment variables
2. Implement token refresh logic for OAuth flows
3. Use HTTPS for all API communications
4. Monitor and log authentication failures
5. Rotate API keys regularly`);
                }
              },
              {
                label: 'News Article Style',
                icon: FileText,
                onClick: () => {
                  onChange(`Local Community Garden Transforms Vacant Lot into Green Oasis

SPRINGFIELD — What was once an eyesore filled with debris and weeds has blossomed into a thriving community garden that brings neighbors together while providing fresh produce to local families.

The Riverside Community Garden, located on the corner of Oak Street and Third Avenue, officially opened its gates six months ago after two years of planning and fundraising efforts led by the Springfield Neighborhood Association.

"When I moved to this neighborhood five years ago, this lot was nothing but broken glass and overgrown weeds," said Maria Rodriguez, 34, who spearheaded the garden initiative. "Now my kids can safely play here while I tend to our tomatoes and peppers."

The half-acre space now features 47 individual garden plots, a shared composting area, a small greenhouse, and a children's learning garden where local schools bring students for hands-on environmental education.

City Council member James Patterson, who supported the project from its inception, attended yesterday's harvest festival. "This garden represents everything that's great about our community," Patterson said. "Neighbors working together, kids learning about where food comes from, and transforming unused space into something beautiful and productive."

The garden operates on a sliding fee scale, with plots ranging from $25 to $75 per season based on family income. Surplus produce is donated to the Springfield Food Bank, which has received over 800 pounds of fresh vegetables since the garden's opening.

"We've had families tell us this is the first time their children have eaten vegetables they actually enjoyed," said volunteer coordinator Susan Chen. "There's something magical about kids eating carrots they planted and watered themselves."

The success has inspired similar projects in neighboring communities. The Westside Residents Association recently broke ground on their own community garden, with plans for three more locations currently in development.

For information about available plots or volunteer opportunities, visit springfieldgarden.org or attend the monthly meetings held the first Saturday of each month at the Springfield Community Center.`);
                }
              }
            ]
          },
          { type: 'separator' as const },
          { 
            label: 'Word Count', 
            icon: FileText, 
            onClick: () => {
              alert(`Words: ${wordCount}, Characters: ${charCount}`);
            }
          }
        ];
        
      case 'actions':
        return [
          {
            label: isAnalyzing ? 'Analyzing...' : 'Analyze Text',
            icon: Search,
            onClick: handleAnalyze,
            disabled: !value.trim() || isAnalyzing
          },
          { type: 'separator' as const },
          {
            label: 'Auto Detection',
            icon: analysisSettings.autoDetection ? ToggleRight : ToggleLeft,
            onClick: () => toggleAnalysisSetting('autoDetection'),
            type: 'toggle' as const,
            checked: analysisSettings.autoDetection
          },
          {
            label: 'Confidence Scoring',
            icon: analysisSettings.showConfidenceScoring ? Eye : EyeOff,
            onClick: () => toggleAnalysisSetting('showConfidenceScoring'),
            type: 'toggle' as const,
            checked: analysisSettings.showConfidenceScoring
          },
          {
            label: 'Detailed Analysis',
            icon: analysisSettings.showDetailedAnalysis ? Eye : EyeOff,
            onClick: () => toggleAnalysisSetting('showDetailedAnalysis'),
            type: 'toggle' as const,
            checked: analysisSettings.showDetailedAnalysis
          }
        ];
      
      case 'format':
        return [
          { 
            label: 'Bold', 
            icon: Bold, 
            onClick: () => applyFormatting('bold'),
            type: 'toggle' as const,
            checked: formatting.bold
          },
          { 
            label: 'Italic', 
            icon: Italic, 
            onClick: () => applyFormatting('italic'),
            type: 'toggle' as const,
            checked: formatting.italic
          },
          { 
            label: 'Underline', 
            icon: Underline, 
            onClick: () => applyFormatting('underline'),
            type: 'toggle' as const,
            checked: formatting.underline
          },
          { type: 'separator' as const },
          { 
            label: 'Bullet List', 
            icon: List, 
            onClick: () => applyFormatting('bullet-list'),
            type: 'toggle' as const,
            checked: formatting.listType === 'bullet'
          },
          { 
            label: 'Numbered List', 
            icon: ListOrdered, 
            onClick: () => applyFormatting('numbered-list'),
            type: 'toggle' as const,
            checked: formatting.listType === 'numbered'
          }
        ];
      
      case 'layout':
        return [
          { 
            label: 'Align Left', 
            icon: AlignLeft, 
            onClick: () => setAlignment('left'),
            type: 'toggle' as const,
            checked: formatting.alignment === 'left'
          },
          { 
            label: 'Align Center', 
            icon: AlignCenter, 
            onClick: () => setAlignment('center'),
            type: 'toggle' as const,
            checked: formatting.alignment === 'center'
          },
          { 
            label: 'Align Right', 
            icon: AlignRight, 
            onClick: () => setAlignment('right'),
            type: 'toggle' as const,
            checked: formatting.alignment === 'right'
          },
          { 
            label: 'Justify', 
            icon: AlignJustify, 
            onClick: () => setAlignment('justify'),
            type: 'toggle' as const,
            checked: formatting.alignment === 'justify'
          }
        ];
      
      case 'watermark':
        return [
          { 
            label: 'Embed Watermark', 
            icon: Shield, 
            onClick: () => {
              const content = prompt('Enter watermark content:', watermarkSettings.content || '© 2024 Your Name');
              if (content !== null) {
                setWatermarkSettings(prev => ({ ...prev, content }));
                if (content.trim()) {
                  embedWatermark();
                }
              }
            },
            disabled: isWatermarking || !value.trim()
          },
          { 
            label: 'Extract Watermark', 
            icon: ShieldCheck, 
            onClick: extractWatermark,
            disabled: isWatermarking || !value.trim()
          },
          { type: 'separator' as const },
          { 
            label: 'Stegano LSB', 
            icon: Eye, 
            onClick: () => setWatermarkSettings(prev => ({ ...prev, method: 'stegano_lsb' })),
            type: 'toggle' as const,
            checked: watermarkSettings.method === 'stegano_lsb'
          },
          { 
            label: 'Visible Text', 
            icon: EyeOff, 
            onClick: () => setWatermarkSettings(prev => ({ ...prev, method: 'visible_text' })),
            type: 'toggle' as const,
            checked: watermarkSettings.method === 'visible_text'
          },
          { type: 'separator' as const },
          { 
            label: 'Export Text', 
            icon: Download, 
            onClick: exportWatermarkedText,
            disabled: !value.trim()
          },
          { 
            label: 'Import Text', 
            icon: Upload, 
            onClick: importText
          }
        ];

      default:
        return [];
    }
  };

  return (
    <div className="text-editor">
      <div className="editor-toolbar">
        <div className="toolbar-left">
          <div className="toolbar-tabs-container">
            <ExpandableTabs 
              tabs={tabs}
              selectedIndex={selectedTabIndex}
              onChange={(index) => {
                console.log('ExpandableTabs onChange called with index:', index);
                if (index === null) {
                  console.log('Closing all dropdowns');
                  setActiveDropdown(null);
                  setSelectedTabIndex(null);
                } else {
                  // Handle the tab selection based on index
                  const tabNames = ['document', 'format', 'layout', 'watermark', null, 'actions']; // null for separator
                  const tabName = tabNames[index];
                  if (tabName) {
                    console.log('Opening dropdown for tab:', tabName);
                    setActiveDropdown(tabName);
                    setSelectedTabIndex(index);
                  }
                }
              }}
            />
            
            {activeDropdown && (
              <DropdownMenu
                items={getDropdownItems(activeDropdown)}
                isOpen={true}
                onClose={() => setActiveDropdown(null)}
                className="toolbar-dropdown"
              />
            )}
          </div>
        </div>
        
        <div className="toolbar-right">
          <div className="word-count">
            {watermarkInfo?.hasWatermark && (
              <span className="count-item" style={{ 
                backgroundColor: '#4ecdc4', 
                color: 'white',
                fontWeight: '600'
              }}>
                🛡️ Watermarked
              </span>
            )}
            <span className="count-item">{wordCount} words</span>
            <span className="count-separator">•</span>
            <span className="count-item">{charCount} characters</span>
          </div>
        </div>
      </div>
      
      <div className="editor-content">
        <div
          ref={editorRef}
          contentEditable
          onInput={handleInput}
          onPaste={handlePaste}
          className={`document-editor align-${formatting.alignment}`}
          spellCheck={true}
          data-placeholder={placeholder}
          suppressContentEditableWarning={true}
        />
      </div>
    </div>
  );
};