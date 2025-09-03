import React from 'react';
import { LucideIcon, ChevronRight } from 'lucide-react';
import './DropdownMenu.css';

interface DropdownItemBase {
  icon?: LucideIcon;
  onClick?: () => void;
  checked?: boolean;
  disabled?: boolean;
}

interface RegularDropdownItem extends DropdownItemBase {
  label: string;
  type?: 'item' | 'toggle';
}

interface SubmenuDropdownItem extends DropdownItemBase {
  label: string;
  type: 'submenu';
  submenu: RegularDropdownItem[];
}

interface SeparatorDropdownItem extends DropdownItemBase {
  label?: never;
  type: 'separator';
}

type DropdownItem = RegularDropdownItem | SubmenuDropdownItem | SeparatorDropdownItem;

interface DropdownMenuProps {
  items: DropdownItem[];
  isOpen: boolean;
  onClose: () => void;
  className?: string;
}

export const DropdownMenu: React.FC<DropdownMenuProps> = ({
  items,
  isOpen,
  onClose,
  className = ""
}) => {
  const menuRef = React.useRef<HTMLDivElement>(null);
  const [activeSubmenu, setActiveSubmenu] = React.useState<number | null>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  console.log('DropdownMenu rendering with items:', items);
  
  return (
    <div ref={menuRef} className={`dropdown-menu ${className}`}>
      {items.map((item, index) => {
        console.log('Processing item:', index, item);
        if (item.type === 'separator') {
          return <div key={index} className="dropdown-separator" />;
        }

        if (item.type === 'submenu') {
          console.log('Found submenu item:', item);
          const Icon = item.icon;
          const isSubmenuActive = activeSubmenu === index;
          console.log('Submenu active state:', isSubmenuActive, 'activeSubmenu:', activeSubmenu, 'index:', index);
          console.log('Submenu items:', item.submenu);
          
          // Debug logging
          if (isSubmenuActive) {
            console.log('Rendering submenu with items:', item.submenu);
          }
          
          return (
            <div key={index} className="dropdown-submenu-container">
              <button
                className={`dropdown-item submenu-trigger ${isSubmenuActive ? 'active' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setActiveSubmenu(isSubmenuActive ? null : index);
                }}
              >
                {Icon && <Icon className="dropdown-item-icon" />}
                <span className="dropdown-item-label">{item.label}</span>
                <ChevronRight className="submenu-arrow" />
              </button>
              
              {isSubmenuActive && (
                <div 
                  className="dropdown-submenu"
                  onClick={(e) => e.stopPropagation()}
                >

                  {item.submenu.map((subItem, subIndex) => {
                    const SubIcon = subItem.icon;
                    return (
                      <button
                        key={subIndex}
                        className={`dropdown-item ${subItem.disabled ? 'disabled' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!subItem.disabled && subItem.onClick) {
                            subItem.onClick();
                          }
                          onClose();
                        }}
                        disabled={subItem.disabled}
                      >
                        {SubIcon && <SubIcon className="dropdown-item-icon" />}
                        <span className="dropdown-item-label">{subItem.label}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        }

        const Icon = item.icon;
        
        return (
          <button
            key={index}
            className={`dropdown-item ${item.disabled ? 'disabled' : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              console.log('Regular dropdown item clicked:', item.label); // Debug log
              alert(`Clicked: ${item.label}`); // Temporary alert for testing
              if (!item.disabled && item.onClick) {
                item.onClick();
              }
            }}
            disabled={item.disabled}
          >
            {Icon && <Icon className="dropdown-item-icon" />}
            <span className="dropdown-item-label">{item.label}</span>
            {item.type === 'toggle' && (
              <div className={`toggle-indicator ${item.checked ? 'checked' : ''}`}>
                {item.checked ? '✓' : ''}
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
};