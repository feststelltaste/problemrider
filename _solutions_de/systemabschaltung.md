---
title: Systemabschaltung
description: Bewusste Außerbetriebnahme von Systemen — mit
  Verantwortlichen, Terminen, einem Datenplan und verifizierter Abschaltung
  — statt sie laufen zu lassen, weil niemand entschieden hat, sie zu
  stoppen.
category:
- Architecture
- Management
- Operations
problems:
- obsolete-technologies
- technology-lock-in
- high-maintenance-costs
- system-stagnation
- vendor-dependency
- technology-stack-fragmentation
- maintenance-cost-increase
- operational-overhead
- knowledge-silos
- increased-cost-of-development
- resource-waste
- monitoring-gaps
- dependency-on-supplier
- lack-of-ownership-and-accountability
- legal-disputes
- modernization-roi-justification-failure
- vendor-dependency-entrapment
- retention-obligations-block-change
layout: solution
lang: de
en_slug: system-decommissioning
related_solutions:
- slug: retention-and-disposal-policy
  similarity: 0.75
- slug: application-portfolio-inventory
  similarity: 0.7
- slug: deprecation-strategy
  similarity: 0.7
- slug: feature-usage-measurement
  similarity: 0.7
- slug: total-cost-of-ownership-transparency
  similarity: 0.65
- slug: backup-and-recovery
  similarity: 0.65
---

## Description

Systemabschaltung ist die bewusste Außerbetriebnahme eines Systems oder einer Komponente: feststellen, dass sie nicht mehr benötigt wird, ihre Daten migrieren oder archivieren, ihre Konsumenten entfernen, sie herunterfahren und verifizieren, dass nichts kaputtging. Sie existiert als benannte Praxis, weil der Standardfall sonst anders ist. Ein System, das noch läuft, ist nie dringend zu stoppen, also läuft es weiter — verbraucht Lizenzen, Infrastruktur, Patching-Aufwand, Monitoring-Aufmerksamkeit und das residuale Wissen der einen Person, die sich erinnert, wie es funktioniert. Legacy-Landschaften sind voll von Systemen, deren Ersatz vor Jahren abgeschlossen wurde und die nie abgeschaltet wurden, weil das Abschalten Risiko trägt und keinen sichtbaren Nutzen bringt. Abschaltung ist die einzige Intervention, die die Gesamtkosten einer Landschaft absolut reduziert, und sie ist systematisch unterfinanziert, weil ihr Nutzen die Abwesenheit von etwas ist.

## How to Apply ◆

> Die Systeme, deren Abschaltung sich am meisten lohnt, sind die, an die niemand denkt, was genau der Grund ist, warum niemand ihre Abschaltung vorschlägt.

- **Stellen Sie fest, wer sie tatsächlich nutzt**, bevor irgendetwas anderes geschieht, unter Nutzung von Evidenz statt Nachfragen. Zugriffsprotokolle, Datenbankverbindungen, Netzwerkverkehr und Authentifizierungsaufzeichnungen sagen Ihnen, was eine Umfrage nicht sagt, weil die Konsumenten, die Sie finden müssen, diejenigen sind, die vergessen haben, dass sie davon abhängen.
- **Weisen Sie einen benannten Verantwortlichen und ein Zieldatum zu.** Eine Abschaltung ohne beides ist eine Aspiration. Der Verantwortliche muss die Arbeit nicht selbst tun, muss aber für ihren Fortschritt zur Verantwortung gezogen werden können.
- **Entscheiden Sie die Datenfrage explizit und früh**, weil sie üblicherweise der schwierige Teil ist: was aufbewahrt werden muss, wie lange, unter welcher rechtlichen oder regulatorischen Verpflichtung, und in welcher Form es in acht Jahren lesbar sein wird. Ein Archiv, das niemand lesen kann, ist keine Aufbewahrungslösung.
- **Migrieren Sie Konsumenten einzeln**, jeder verifiziert, statt ein Abschaltdatum anzukündigen und zu erwarten, dass sie umziehen. Konsumenten, die die Frist nicht einhalten, sind üblicherweise diejenigen, die nie wussten, dass sie Konsumenten waren.
- **Ankündigen, dann beobachten.** Lassen Sie das System nach der Migration des letzten bekannten Konsumenten mit Monitoring auf allen Zugriffen für einen vollen Geschäftszyklus laufen. Alles, was in diesem Fenster erscheint, ist ein Konsument, den Sie nicht gefunden haben, und ihn hier zu finden ist weit günstiger als nach der Abschaltung.
- **Schalten Sie zuerst auf reversible Weise ab** — deaktivieren Sie Zugriff, stoppen Sie den Dienst, behalten Sie die Daten und die Fähigkeit zum Neustart — und nehmen Sie erst dann die Infrastruktur außer Betrieb. Die Lücke zwischen Stoppen und Löschen ist die Sicherheitsmarge.
- **Decken Sie die vollständige Fläche ab** beim Entfernen: geplante Jobs, Monitoring und Alarme, Firewall-Regeln, DNS-Einträge, Zugangsdaten, Service-Konten, Backup-Jobs, Lizenzen und Support-Verträge. Halb abgeschaltete Systeme erzeugen Alarme, die niemandem gehören, und hinterlassen Zugangsdaten, die niemand rotiert, was sowohl betriebliches Rauschen als auch eine Sicherheitsexposition ist.
- **Erfassen Sie, was das System wusste.** Geschäftsregeln, die nur in einem außer Betrieb genommenen System kodiert sind, gehen verloren, wenn es geht, und dieser Verlust wird häufig erst Jahre später entdeckt. Dokumentieren Sie die Regeln, oder verifizieren Sie, dass sie im Ersatz implementiert sind.
- **Protokollieren Sie die Einsparung.** Beendete Lizenzen, freigegebene Infrastruktur, vermiedener Patching-Aufwand. Eine Abschaltung, deren Nutzen nie quantifiziert wird, macht die nächste schwerer zu finanzieren, und die angesammelte Zahl ist das Argument für ein stehendes Ausmusterungsprogramm.
- **Pflegen Sie eine Kandidatenliste** mit dem Inventar dessen, was existiert, periodisch überprüft. Ausmusterung geschieht, wenn sie der ständige Tagesordnungspunkt von jemandem ist, statt wenn eine Krise sie erzwingt.

## Tradeoffs ⇄

> Abschaltung ist die einzige Änderung, die die Gesamtsystemkosten direkt senkt, trägt aber echtes Risiko, liefert kein sichtbares Feature, und der Aufwand ist häufig mit dem Bau von etwas Neuem vergleichbar.

**Vorteile:**

- Kosten fallen absolut und dauerhaft — Lizenzen, Infrastruktur, Support-Verträge und der Patching- und Monitoring-Aufwand, den das System verbrauchte.
- Die Anzahl der Technologien, für die die Organisation Fähigkeiten aufrechterhalten muss, sinkt, was der Engpass hinter einem Großteil der Brüchigkeit einer Legacy-Landschaft ist.
- Sicherheitsexposition schrumpft, da die Systeme, die am wahrscheinlichsten ausgemustert werden, auch die am wenigsten wahrscheinlich gepatchten sind.
- Aufmerksamkeit wird freigesetzt. Jedes laufende System belegt einen Anteil an Monitoring-, Bereitschafts- und Audit-Aufwand, unabhängig davon, ob es jemand nutzt.
- Die verbleibende Landschaft wird verständlich, was jede nachfolgende Auswirkungsanalyse und Modernisierungsschätzung verbessert.

**Kosten und Risiken:**

- Das Herunterfahren eines Systems mit einem unentdeckten Konsumenten verursacht einen Fehler ohne offensichtliche Ursache, da die entfernte Abhängigkeit per Definition unsichtbar ist.
- Datenaufbewahrungsverpflichtungen können genuin komplex sein, und sie falsch zu machen ist eher eine rechtliche als eine betriebliche Exposition.
- Der Aufwand ist oft erheblich und liefert nichts Sichtbares, was ihn schwer zu finanzieren macht gegen Arbeit, die Features produziert.
- Geschäftsregeln, die nur im ausgemusterten System kodiert sind, können verloren gehen, und der Verlust taucht lange auf, nachdem das Wissen zu ihrer Wiederherstellung verschwunden ist.
- Teilweise Abschaltung ist schlimmer als keine: verwaiste Alarme, nicht rotierte Zugangsdaten und Infrastruktur, die niemandem gehört, häufen sich als eigene Kategorie von Schulden an.

## How It Could Be

Das Anwendungsinventar einer Organisation listete 84 Systeme auf. Die Untersuchung von Zugriffsprotokollen ergab, dass 11 seit über sechs Monaten keinen aufgezeichneten menschlichen oder Systemzugriff hatten, einschließlich eines Berichtswerkzeugs, das drei Jahre zuvor ersetzt worden war und dessen Nachfolger die ganze Zeit im Einsatz gewesen war. Niemand hatte das Original abgeschaltet, weil dies verlangte, dass jemand zuversichtlich war, und niemand war es. Eine Abschaltungsanstrengung mit einem benannten Verantwortlichen und Terminen arbeitete über zwei Quartale hinweg neun der 11 ab, nach dem Muster beobachten-dann-deaktivieren-dann-löschen. Zwei brachten Überraschungen während des Beobachtungsfensters hervor: eines wurde noch nächtlich von einer Partnerintegration aufgerufen, die niemand dokumentiert hatte, und eines enthielt die einzige Kopie von sieben Jahren Audit-Aufzeichnungen, die einer Aufbewahrungsverpflichtung unterlagen. Beide wurden vor statt nach der Abschaltung gelöst. Die direkte jährliche Einsparung bei Lizenzen und Infrastruktur reichte, um den Aufwand ungefähr dreimal zu finanzieren, und die Bereitschaftsrotation verlor vier Alarmquellen, die routinemäßig ignoriert worden waren.

Der Wissenserfassungsschritt rechtfertigte sich beim zehnten System. Ein außer Betrieb genommenes Batch-Planungswerkzeug enthielt die Abhängigkeitsreihenfolge für etwa 60 nächtliche Jobs, kodiert in seiner Konfiguration und nirgendwo sonst. Der Ersatz war konfiguriert worden, indem die scheinbar relevanten Einträge kopiert wurden, und ein Review während der Abschaltung fand vier Reihenfolgeabhängigkeiten, die nicht übertragen worden waren — keine davon hatte bisher einen Fehler verursacht, weil das Timing an den meisten Abenden zufällig aufging. Diese zu entdecken, während das Original noch zur Konsultation verfügbar war, dauerte zwei Tage. Sie nach der Abschaltung zu entdecken hätte bedeutet, intermittierende Dateninkonsistenzen zu diagnostizieren, während die maßgebliche Quelle verschwunden war.
