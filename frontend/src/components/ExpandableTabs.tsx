import React from 'react';
import { LucideIcon } from 'lucide-react';

interface Tab {
  title: string;
  icon: LucideIcon;
  type?: never;
  onClick?: () => void;
}

interface Separator {
  type: "separator";
  title?: never;
  icon?: never;
  onClick?: never;
}

type TabItem = Tab | Separator;

interface ExpandableTabsProps {
  tabs: TabItem[];
  className?: string;
  activeColor?: string;
  onChange?: (index: number | null) => void;
  selectedIndex?: number | null;
}

export const ExpandableTabs: React.FC<ExpandableTabsProps> = ({
  tabs,
  className = "",
  activeColor = "text-blue-500",
  onChange,
  selectedIndex,
}) => {
  const [selected, setSelected] = React.useState<number | null>(selectedIndex || null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    setSelected(selectedIndex || null);
  }, [selectedIndex]);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      // Don't close if clicking on dropdown menu
      const target = event.target as Element;
      if (target.closest('.dropdown-menu')) {
        return;
      }
      
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setSelected(null);
        onChange?.(null);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onChange]);

  const handleSelect = (index: number, tab: Tab) => {
    const newSelected = selected === index ? null : index;
    setSelected(newSelected);
    onChange?.(newSelected);
    
    // Call the tab's onClick handler
    if (tab.onClick) {
      tab.onClick();
    }
  };

  const Separator = () => (
    <div className="separator" />
  );

  return (
    <div
      ref={containerRef}
      className={`expandable-tabs ${className}`}
    >
      {tabs.map((tab, index) => {
        if (tab.type === "separator") {
          return <Separator key={`separator-${index}`} />;
        }

        const Icon = tab.icon;
        const isSelected = selected === index;
        
        return (
          <button
            key={tab.title}
            onClick={() => handleSelect(index, tab)}
            className={`expandable-tab ${isSelected ? 'selected' : ''}`}
          >
            <Icon className="tab-icon" />
            {isSelected && (
              <span className="tab-title">{tab.title}</span>
            )}
          </button>
        );
      })}
    </div>
  );
};